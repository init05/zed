use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, anyhow};
use fs::{Fs, RemoveOptions, RenameOptions};
use gpui::{App, AsyncApp, Entity, Task};
use project::{
    LocalProjectFlags, Project, WorktreeId,
    git_store::{Repository, worktrees_directory_for_repo},
    project_settings::ProjectSettings,
};
use remote::{RemoteConnectionOptions, same_remote_connection_identity};
use settings::Settings;
use util::ResultExt;
use workspace::{AppState, MultiWorkspace, Workspace};

use crate::thread_metadata_store::{ArchivedGitWorktree, ThreadId, ThreadMetadataStore};

/// Controls whether [`restore_worktree_via_git`] should proceed when
/// pre-existing content is found at the worktree path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverwritePolicy {
    /// Return an error if the worktree path has content, without doing
    /// any destructive work. Callers use this for a read-only preflight.
    Refuse,
    /// Move pre-existing content to a backup and proceed with the restore.
    Overwrite,
}

/// The plan for archiving a single git worktree root.
///
/// A thread can have multiple folder paths open, so there may be multiple
/// `RootPlan`s per archival operation. Each one captures everything needed to
/// persist the worktree's git state and then remove it from disk.
///
/// All fields are gathered synchronously by [`build_root_plan`] while the
/// worktree is still loaded in open projects. This is important because
/// workspace removal tears down project and repository entities, making
/// them unavailable for the later async persist/remove steps.
#[derive(Clone)]
pub struct RootPlan {
    /// Absolute path of the git worktree on disk.
    pub root_path: PathBuf,
    /// Absolute path to the main git repository this worktree is linked to.
    /// Used both for creating a git ref to prevent GC of WIP commits during
    /// [`persist_worktree_state`], and for `git worktree remove` during
    /// [`remove_root`].
    pub main_repo_path: PathBuf,
    /// Every open `Project` that has this worktree loaded, so they can all
    /// call `remove_worktree` and release it during [`remove_root`].
    /// Multiple projects can reference the same path when the user has the
    /// worktree open in more than one workspace.
    pub affected_projects: Vec<AffectedProject>,
    /// The `Repository` entity for this linked worktree, used to run git
    /// commands (create WIP commits, stage files, reset) during
    /// [`persist_worktree_state`].
    pub worktree_repo: Entity<Repository>,
    /// The branch the worktree was on, so it can be restored later.
    /// `None` if the worktree was in detached HEAD state.
    pub branch_name: Option<String>,
    /// Remote connection options for the project that owns this worktree,
    /// used to create temporary remote projects when the main repo isn't
    /// loaded in any open workspace.
    pub remote_connection: Option<RemoteConnectionOptions>,
}

/// A `Project` that references a worktree being archived, paired with the
/// `WorktreeId` it uses for that worktree.
///
/// The same worktree path can appear in multiple open workspaces/projects
/// (e.g. when the user has two windows open that both include the same
/// linked worktree). Each one needs to call `remove_worktree` and wait for
/// the release during [`remove_root`], otherwise the project would still
/// hold a reference to the directory and `git worktree remove` would fail.
#[derive(Clone)]
pub struct AffectedProject {
    pub project: Entity<Project>,
    pub worktree_id: WorktreeId,
}

fn archived_worktree_ref_name(id: i64) -> String {
    format!("refs/archived-worktrees/{}", id)
}

/// Resolves the Zed-managed worktrees base directory for a given repo.
///
/// This intentionally reads the *global* `git.worktree_directory` setting
/// rather than any project-local override, because Zed always uses the
/// global value when creating worktrees and the archive check must match.
fn worktrees_base_for_repo(main_repo_path: &Path, cx: &App) -> Option<PathBuf> {
    let setting = &ProjectSettings::get_global(cx).git.worktree_directory;
    worktrees_directory_for_repo(main_repo_path, setting).log_err()
}

/// Builds a [`RootPlan`] for archiving the git worktree at `path`.
///
/// This is a synchronous planning step that must run *before* any workspace
/// removal, because it needs live project and repository entities that are
/// torn down when a workspace is removed. It does three things:
///
/// 1. Finds every `Project` across all open workspaces that has this
///    worktree loaded (`affected_projects`).
/// 2. Looks for a `Repository` entity whose snapshot identifies this path
///    as a linked worktree (`worktree_repo`), which is needed for the git
///    operations in [`persist_worktree_state`].
/// 3. Determines the `main_repo_path` — the parent repo that owns this
///    linked worktree — needed for both git ref creation and
///    `git worktree remove`.
///
/// Returns `None` if the path is not a linked worktree (main worktrees
/// cannot be archived to disk) or if no open project has it loaded.
pub fn build_root_plan(
    path: &Path,
    remote_connection: Option<&RemoteConnectionOptions>,
    workspaces: &[Entity<Workspace>],
    cx: &App,
) -> Option<RootPlan> {
    let path = path.to_path_buf();

    let matches_target_connection = |project: &Entity<Project>, cx: &App| {
        same_remote_connection_identity(
            project.read(cx).remote_connection_options(cx).as_ref(),
            remote_connection,
        )
    };

    let affected_projects = workspaces
        .iter()
        .filter_map(|workspace| {
            let project = workspace.read(cx).project().clone();
            if !matches_target_connection(&project, cx) {
                return None;
            }
            let worktree = project
                .read(cx)
                .visible_worktrees(cx)
                .find(|worktree| worktree.read(cx).abs_path().as_ref() == path.as_path())?;
            let worktree_id = worktree.read(cx).id();
            Some(AffectedProject {
                project,
                worktree_id,
            })
        })
        .collect::<Vec<_>>();

    if affected_projects.is_empty() {
        return None;
    }

    let linked_repo = workspaces
        .iter()
        .filter(|workspace| matches_target_connection(workspace.read(cx).project(), cx))
        .flat_map(|workspace| {
            workspace
                .read(cx)
                .project()
                .read(cx)
                .repositories(cx)
                .values()
                .cloned()
                .collect::<Vec<_>>()
        })
        .find_map(|repo| {
            let snapshot = repo.read(cx).snapshot();
            (snapshot.is_linked_worktree()
                && snapshot.work_directory_abs_path.as_ref() == path.as_path())
            .then_some((snapshot, repo))
        });

    // Only linked worktrees can be archived to disk via `git worktree remove`.
    // Main worktrees must be left alone — git refuses to remove them.
    let (linked_snapshot, repo) = linked_repo?;
    let main_repo_path = linked_snapshot.main_worktree_abs_path()?.to_path_buf();

    // Only archive worktrees that live inside the Zed-managed worktrees
    // directory (configured via `git.worktree_directory`). Worktrees the
    // user created outside that directory should be left untouched.
    let worktrees_base = worktrees_base_for_repo(&main_repo_path, cx)?;
    if !path.starts_with(&worktrees_base) {
        return None;
    }

    let branch_name = linked_snapshot
        .branch
        .as_ref()
        .map(|branch| branch.name().to_string());

    Some(RootPlan {
        root_path: path,
        main_repo_path,
        affected_projects,
        worktree_repo: repo,
        branch_name,
        remote_connection: remote_connection.cloned(),
    })
}

/// Removes a worktree from all affected projects and deletes it from disk
/// via `git worktree remove`.
///
/// This is the destructive counterpart to [`persist_worktree_state`]. It
/// first detaches the worktree from every [`AffectedProject`], waits for
/// each project to fully release it, then asks the main repository to
/// delete the worktree directory. If the git removal fails, the worktree
/// is re-added to each project via [`rollback_root`].
pub async fn remove_root(root: RootPlan, cx: &mut AsyncApp) -> Result<()> {
    let release_tasks: Vec<_> = root
        .affected_projects
        .iter()
        .map(|affected| {
            let project = affected.project.clone();
            let worktree_id = affected.worktree_id;
            project.update(cx, |project, cx| {
                let wait = project.wait_for_worktree_release(worktree_id, cx);
                project.remove_worktree(worktree_id, cx);
                wait
            })
        })
        .collect();

    if let Err(error) = remove_root_after_worktree_removal(&root, release_tasks, cx).await {
        rollback_root(&root, cx).await;
        return Err(error);
    }

    Ok(())
}

async fn remove_root_after_worktree_removal(
    root: &RootPlan,
    release_tasks: Vec<Task<Result<()>>>,
    cx: &mut AsyncApp,
) -> Result<()> {
    for task in release_tasks {
        if let Err(error) = task.await {
            log::error!("Failed waiting for worktree release: {error:#}");
        }
    }

    let (repo, project) =
        find_or_create_repository(&root.main_repo_path, root.remote_connection.as_ref(), cx)
            .await?;

    // `Repository::remove_worktree` with `force = true` deletes the working
    // directory before running `git worktree remove --force`, so there's no
    // need to touch the filesystem here. For remote projects that cleanup
    // runs on the headless server via the `GitRemoveWorktree` RPC, which is
    // the only code path with access to the remote machine's filesystem.
    let receiver = repo.update(cx, |repo: &mut Repository, _cx| {
        repo.remove_worktree(root.root_path.clone(), true)
    });
    let result = receiver
        .await
        .map_err(|_| anyhow!("git worktree metadata cleanup was canceled"))?;
    // `project` may be a live workspace project or a temporary one created
    // by `find_or_create_repository`. In the temporary case we must keep it
    // alive until the repo removes the worktree
    drop(project);
    result.context("git worktree metadata cleanup failed")?;
    Ok(())
}

/// Finds a live `Repository` entity for the given path, or creates a temporary
/// project to obtain one.
///
/// `Repository` entities can only be obtained through a `Project` because
/// `GitStore` (which creates and manages `Repository` entities) is owned by
/// `Project`. When no open workspace contains the repo we need, we spin up a
/// headless project just to get a `Repository` handle. For local paths this is
/// a `Project::local`; for remote paths we build a `Project::remote` through
/// the connection pool (reusing the existing SSH transport), which requires
/// the caller to pass the matching `RemoteConnectionOptions` so we only match
/// and fall back onto projects that share the same remote identity. The
/// caller keeps the returned `Entity<Project>` alive for the duration of the
/// git operations, then drops it.
///
/// Future improvement: decoupling `GitStore` from `Project` so that
/// `Repository` entities can be created standalone would eliminate this
/// temporary-project workaround.
async fn find_or_create_repository(
    repo_path: &Path,
    remote_connection: Option<&RemoteConnectionOptions>,
    cx: &mut AsyncApp,
) -> Result<(Entity<Repository>, Entity<Project>)> {
    let repo_path_owned = repo_path.to_path_buf();
    let remote_connection_owned = remote_connection.cloned();

    // First, try to find a live repository in any open workspace whose
    // remote connection matches (so a local `/project` and a remote
    // `/project` are not confused).
    let live_repo = cx.update(|cx| {
        all_open_workspaces(cx)
            .into_iter()
            .filter_map(|workspace| {
                let project = workspace.read(cx).project().clone();
                let project_connection = project.read(cx).remote_connection_options(cx);
                if !same_remote_connection_identity(
                    project_connection.as_ref(),
                    remote_connection_owned.as_ref(),
                ) {
                    return None;
                }
                Some((
                    project
                        .read(cx)
                        .repositories(cx)
                        .values()
                        .find(|repo| {
                            repo.read(cx).snapshot().work_directory_abs_path.as_ref()
                                == repo_path_owned.as_path()
                        })
                        .cloned()?,
                    project.clone(),
                ))
            })
            .next()
    });

    if let Some((repo, project)) = live_repo {
        return Ok((repo, project));
    }

    let app_state =
        current_app_state(cx).context("no app state available for temporary project")?;

    // For remote paths, create a fresh RemoteClient through the connection
    // pool (reusing the existing SSH transport) and build a temporary
    // remote project. Each RemoteClient gets its own server-side headless
    // project, so there are no RPC routing conflicts with other projects.
    let temp_project = if let Some(connection) = remote_connection_owned {
        let remote_client = cx
            .update(|cx| {
                if !remote::has_active_connection(&connection, cx) {
                    anyhow::bail!("cannot open repository on disconnected remote machine");
                }
                Ok(remote_connection::connect_reusing_pool(connection, cx))
            })?
            .await?
            .context("remote connection was canceled")?;

        cx.update(|cx| {
            Project::remote(
                remote_client,
                app_state.client.clone(),
                app_state.node_runtime.clone(),
                app_state.user_store.clone(),
                app_state.languages.clone(),
                app_state.fs.clone(),
                false,
                cx,
            )
        })
    } else {
        cx.update(|cx| {
            Project::local(
                app_state.client.clone(),
                app_state.node_runtime.clone(),
                app_state.user_store.clone(),
                app_state.languages.clone(),
                app_state.fs.clone(),
                None,
                LocalProjectFlags::default(),
                cx,
            )
        })
    };

    let repo_path_for_worktree = repo_path.to_path_buf();
    let create_worktree = temp_project.update(cx, |project, cx| {
        project.create_worktree(repo_path_for_worktree, true, cx)
    });
    let _worktree = create_worktree.await?;
    let initial_scan = temp_project.read_with(cx, |project, cx| project.wait_for_initial_scan(cx));
    initial_scan.await;

    let repo_path_for_find = repo_path.to_path_buf();
    let repo = temp_project
        .update(cx, |project, cx| {
            project
                .repositories(cx)
                .values()
                .find(|repo| {
                    repo.read(cx).snapshot().work_directory_abs_path.as_ref()
                        == repo_path_for_find.as_path()
                })
                .cloned()
        })
        .context("failed to resolve temporary repository handle")?;

    let barrier = repo.update(cx, |repo: &mut Repository, _cx| repo.barrier());
    barrier
        .await
        .map_err(|_| anyhow!("temporary repository barrier canceled"))?;
    Ok((repo, temp_project))
}

/// Re-adds the worktree to every affected project after a failed
/// [`remove_root`].
async fn rollback_root(root: &RootPlan, cx: &mut AsyncApp) {
    for affected in &root.affected_projects {
        let task = affected.project.update(cx, |project, cx| {
            project.create_worktree(root.root_path.clone(), true, cx)
        });
        task.await.log_err();
    }
}

/// Saves the worktree's full git state so it can be restored later.
///
/// This creates two detached commits (via [`create_archive_checkpoint`] on
/// the `GitRepository` trait) that capture the staged and unstaged state
/// without moving any branch ref. The commits are:
///   - "WIP staged": a tree matching the current index, parented on HEAD
///   - "WIP unstaged": a tree with all files (including untracked),
///     parented on the staged commit
///
/// After creating the commits, this function:
///   1. Records the commit SHAs, branch name, and paths in a DB record.
///   2. Links every thread referencing this worktree to that record.
///   3. Creates a git ref on the main repo to prevent GC of the commits.
///
/// On success, returns the archived worktree DB row ID for rollback.
pub async fn persist_worktree_state(root: &RootPlan, cx: &mut AsyncApp) -> Result<i64> {
    let worktree_repo = root.worktree_repo.clone();

    let original_commit_hash = worktree_repo
        .update(cx, |repo, _cx| repo.head_sha())
        .await
        .map_err(|_| anyhow!("head_sha canceled"))?
        .context("failed to read original HEAD SHA")?
        .context("HEAD SHA is None")?;

    // Create two detached WIP commits without moving the branch.
    let checkpoint_rx = worktree_repo.update(cx, |repo, _cx| repo.create_archive_checkpoint());
    let (staged_commit_hash, unstaged_commit_hash) = checkpoint_rx
        .await
        .map_err(|_| anyhow!("create_archive_checkpoint canceled"))?
        .context("failed to create archive checkpoint")?;

    // Create DB record
    let store = cx.update(|cx| ThreadMetadataStore::global(cx));
    let worktree_path_str = root.root_path.to_string_lossy().to_string();
    let main_repo_path_str = root.main_repo_path.to_string_lossy().to_string();
    let branch_name = root.branch_name.clone().or_else(|| {
        worktree_repo.read_with(cx, |repo, _cx| {
            repo.snapshot()
                .branch
                .as_ref()
                .map(|branch| branch.name().to_string())
        })
    });

    let db_result = store
        .read_with(cx, |store, cx| {
            store.create_archived_worktree(
                worktree_path_str.clone(),
                main_repo_path_str.clone(),
                branch_name.clone(),
                staged_commit_hash.clone(),
                unstaged_commit_hash.clone(),
                original_commit_hash.clone(),
                cx,
            )
        })
        .await
        .context("failed to create archived worktree DB record");
    let archived_worktree_id = match db_result {
        Ok(id) => id,
        Err(error) => {
            return Err(error);
        }
    };

    // Link all threads on this worktree to the archived record
    let thread_ids: Vec<ThreadId> = store.read_with(cx, |store, _cx| {
        store
            .entries()
            .filter(|thread| {
                thread
                    .folder_paths()
                    .paths()
                    .iter()
                    .any(|p| p.as_path() == root.root_path)
            })
            .map(|thread| thread.thread_id)
            .collect()
    });

    for thread_id in &thread_ids {
        let link_result = store
            .read_with(cx, |store, cx| {
                store.link_thread_to_archived_worktree(*thread_id, archived_worktree_id, cx)
            })
            .await;
        if let Err(error) = link_result {
            if let Err(delete_error) = store
                .read_with(cx, |store, cx| {
                    store.delete_archived_worktree(archived_worktree_id, cx)
                })
                .await
            {
                log::error!(
                    "Failed to delete archived worktree DB record during link rollback: \
                     {delete_error:#}"
                );
            }
            return Err(error.context("failed to link thread to archived worktree"));
        }
    }

    // Create git ref on main repo to prevent GC of the detached commits.
    // This is fatal: without the ref, git gc will eventually collect the
    // WIP commits and a later restore will silently fail.
    let ref_name = archived_worktree_ref_name(archived_worktree_id);
    let (main_repo, _temp_project) =
        find_or_create_repository(&root.main_repo_path, root.remote_connection.as_ref(), cx)
            .await
            .context("could not open main repo to create archive ref")?;
    let rx = main_repo.update(cx, |repo, _cx| {
        repo.update_ref(ref_name.clone(), unstaged_commit_hash.clone())
    });
    rx.await
        .map_err(|_| anyhow!("update_ref canceled"))
        .and_then(|r| r)
        .with_context(|| format!("failed to create ref {ref_name} on main repo"))?;
    // See note in `remove_root_after_worktree_removal`: this may be a live
    // or temporary project; dropping only matters in the temporary case.
    drop(_temp_project);

    Ok(archived_worktree_id)
}

/// Undoes a successful [`persist_worktree_state`] by deleting the git ref
/// on the main repo and removing the DB record. Since the WIP commits are
/// detached (they don't move any branch), no git reset is needed — the
/// commits will be garbage-collected once the ref is removed.
pub async fn rollback_persist(archived_worktree_id: i64, root: &RootPlan, cx: &mut AsyncApp) {
    // Delete the git ref on main repo
    if let Ok((main_repo, _temp_project)) =
        find_or_create_repository(&root.main_repo_path, root.remote_connection.as_ref(), cx).await
    {
        let ref_name = archived_worktree_ref_name(archived_worktree_id);
        let rx = main_repo.update(cx, |repo, _cx| repo.delete_ref(ref_name));
        rx.await.ok().and_then(|r| r.log_err());
        // See note in `remove_root_after_worktree_removal`: this may be a
        // live or temporary project; dropping only matters in the temporary
        // case.
        drop(_temp_project);
    }

    // Delete the DB record
    let store = cx.update(|cx| ThreadMetadataStore::global(cx));
    if let Err(error) = store
        .read_with(cx, |store, cx| {
            store.delete_archived_worktree(archived_worktree_id, cx)
        })
        .await
    {
        log::error!("Failed to delete archived worktree DB record during rollback: {error:#}");
    }
}

/// Restores a previously archived worktree back to disk from its DB record.
///
/// Creates the git worktree at the original commit (the branch never moved
/// during archival since WIP commits are detached), switches to the branch,
/// then uses [`restore_archive_checkpoint`] to reconstruct the staged/
/// unstaged state from the WIP commit trees.
///
/// **Destructive**: the final step (`restore_archive_checkpoint`) clobbers the
/// working directory unconditionally via `git read-tree --reset -u`. Any
/// pre-existing entry at `worktree_path` is moved aside into a
/// `zed-restore-backup-<uuid>` directory under the system temp directory
/// before the rest of the destructive
/// work runs. If a later step fails, the backup is moved back over
/// `worktree_path` so the user does not lose their content. On success the
/// backup directory is deleted.
///
/// The `overwrite_policy` parameter controls behaviour when pre-existing
/// content is found at `worktree_path`:
///
/// * [`OverwritePolicy::Refuse`] — returns an error without doing any
///   destructive work, suitable for a preflight check.
/// * [`OverwritePolicy::Overwrite`] — moves the content to a backup and
///   proceeds with the restore.
pub async fn restore_worktree_via_git(
    row: &ArchivedGitWorktree,
    remote_connection: Option<&RemoteConnectionOptions>,
    overwrite_policy: OverwritePolicy,
    cx: &mut AsyncApp,
) -> Result<PathBuf> {
    if remote_connection.is_some() {
        anyhow::bail!("restoring archived worktrees on remote machines is not yet supported");
    }
    let app_state = current_app_state(cx).context("no app state available")?;
    let worktree_path = &row.worktree_path;

    let (main_repo, _temp_project) =
        find_or_create_repository(&row.main_repo_path, remote_connection, cx).await?;

    // Always restore by recreating the worktree from scratch. This collapses
    // every messy intermediate state into one clean flow:
    //
    //   - Path missing, no registration:               plain add.
    //   - Path missing, stale registration:            scoped remove → add.
    //   - Path present (any kind), no registration
    //     (the original Windows file-lock bug):        rename → add.
    //   - Path present, stale registration:            rename → scoped remove → add.
    //   - Path present as a fully valid worktree:      rename → scoped remove → add.
    //
    // Any pre-existing entry at `worktree_path` is moved aside into a
    // sibling backup directory rather than deleted up-front. If any of the
    // destructive steps below fail, [`rollback_backup`] restores the
    // backup over `worktree_path` so the user does not lose content they
    // confirmed they wanted overwritten only on the assumption that the
    // archived state would replace it. On success the backup is deleted at
    // the end of this function.
    let path_exists = app_state.fs.metadata(worktree_path).await?.is_some();

    if path_exists
        && overwrite_policy == OverwritePolicy::Refuse
        && worktree_path_has_content(app_state.fs.as_ref(), worktree_path).await?
    {
        anyhow::bail!(
            "worktree path '{}' has existing content; use OverwritePolicy::Overwrite to proceed",
            worktree_path.display()
        );
    }

    let backup = if path_exists {
        let backup_dir =
            std::env::temp_dir().join(format!("zed-restore-backup-{}", uuid::Uuid::new_v4()));
        app_state
            .fs
            .create_dir(&backup_dir)
            .await
            .with_context(|| {
                format!(
                    "failed to create backup directory '{}'",
                    backup_dir.display()
                )
            })?;
        let backup_target = backup_dir.join("worktree");
        // `rename` works for both directories and files, so we don't need
        // to dispatch on the entry kind. A stray regular file or symlink
        // at `worktree_path` is moved aside the same way as a directory.
        app_state
            .fs
            .rename(
                worktree_path,
                &backup_target,
                RenameOptions {
                    overwrite: false,
                    ignore_if_exists: false,
                    create_parents: false,
                },
            )
            .await
            .with_context(|| {
                format!(
                    "failed to move existing path '{}' to backup '{}'",
                    worktree_path.display(),
                    backup_target.display()
                )
            })?;
        Some(Backup {
            dir: backup_dir,
            target: backup_target,
        })
    } else {
        None
    };

    // Clean up any stale registration in the main repo that points at the
    // (now missing) worktree path. Without this, `git worktree add` would
    // fail with "already assigned but missing".
    //
    // We call `git worktree remove --force` scoped to this exact path
    // rather than `git worktree prune`, which would have the side effect
    // of dropping admin entries for *other* unrelated worktrees whose
    // working directories happen to be missing. If there is no
    // registration to remove, `git worktree remove` exits with "is not a
    // working tree"; we treat that as a successful no-op since the
    // subsequent `git worktree add` will surface any real configuration
    // problem on its own.
    let remove_rx = main_repo.update(cx, |repo, _cx| {
        repo.remove_worktree(worktree_path.clone(), true)
    });
    if let Err(error) = remove_rx
        .await
        .map_err(|_| anyhow!("worktree remove was canceled"))
        .and_then(|r| r)
    {
        let error_message = format!("{error:#}");
        if error_message.contains("not a working tree") || error_message.contains("no worktree") {
            log::debug!(
                "git worktree remove --force for '{}' (no stale registration): {error:#}",
                worktree_path.display()
            );
        } else {
            rollback_backup(
                app_state.fs.as_ref(),
                backup.as_ref(),
                worktree_path,
                &error,
            )
            .await;
            return Err(error.context(format!(
                "failed to remove stale worktree registration for '{}'",
                worktree_path.display()
            )));
        }
    }

    // Create the worktree at the original commit — the branch still points
    // here because archival used detached commits.
    let create_rx = main_repo.update(cx, |repo, _cx| {
        repo.create_worktree_detached(worktree_path.clone(), row.original_commit_hash.clone())
    });
    let create_result = match create_rx.await {
        Ok(result) => result.context("failed to create worktree"),
        Err(_) => Err(anyhow!("worktree creation was canceled")),
    };
    if let Err(error) = create_result {
        // `create_worktree_detached` may have left a partial directory or
        // a stale registration behind; force-remove first so the rollback
        // rename has somewhere to put the backup back.
        remove_new_worktree_on_error(&main_repo, worktree_path, cx).await;
        rollback_backup(
            app_state.fs.as_ref(),
            backup.as_ref(),
            worktree_path,
            &error,
        )
        .await;
        return Err(error);
    }

    let (wt_repo, _temp_wt_project) =
        match find_or_create_repository(worktree_path, remote_connection, cx).await {
            Ok(result) => result,
            Err(error) => {
                remove_new_worktree_on_error(&main_repo, worktree_path, cx).await;
                rollback_backup(
                    app_state.fs.as_ref(),
                    backup.as_ref(),
                    worktree_path,
                    &error,
                )
                .await;
                return Err(error);
            }
        };

    if let Some(branch_name) = &row.branch_name {
        // Attempt to check out the branch the worktree was previously on.
        let checkout_result = wt_repo
            .update(cx, |repo, _cx| repo.change_branch(branch_name.clone()))
            .await;

        match checkout_result.map_err(|e| anyhow!("{e}")).flatten() {
            Ok(()) => {
                // Branch checkout succeeded. Check whether the branch has moved since
                // we archived the worktree, by comparing HEAD to the expected SHA.
                let head_sha = wt_repo
                    .update(cx, |repo, _cx| repo.head_sha())
                    .await
                    .map_err(|e| anyhow!("{e}"))
                    .and_then(|r| r);

                match head_sha {
                    Ok(Some(sha)) if sha == row.original_commit_hash => {
                        // Branch still points at the original commit; we're all done!
                    }
                    Ok(Some(sha)) => {
                        // The branch has moved. We don't want to restore the worktree to
                        // a different filesystem state, so checkout the original commit
                        // in detached HEAD state.
                        log::info!(
                            "Branch '{branch_name}' has moved since archival (now at {sha}); \
                             restoring worktree in detached HEAD at {}",
                            row.original_commit_hash
                        );
                        let detach_result = main_repo
                            .update(cx, |repo, _cx| {
                                repo.checkout_branch_in_worktree(
                                    row.original_commit_hash.clone(),
                                    row.worktree_path.clone(),
                                    false,
                                )
                            })
                            .await;

                        if let Err(error) = detach_result.map_err(|e| anyhow!("{e}")).flatten() {
                            log::warn!(
                                "Failed to detach HEAD at {}: {error:#}",
                                row.original_commit_hash
                            );
                        }
                    }
                    Ok(None) => {
                        log::warn!(
                            "head_sha unexpectedly returned None after checking out \"{branch_name}\"; \
                             proceeding in current HEAD state."
                        );
                    }
                    Err(error) => {
                        log::warn!(
                            "Failed to read HEAD after checking out \"{branch_name}\": {error:#}"
                        );
                    }
                }
            }
            Err(checkout_error) => {
                // We weren't able to check out the branch, most likely because it was deleted.
                // This is fine; users will often delete old branches! We'll try to recreate it.
                log::debug!(
                    "change_branch('{branch_name}') failed: {checkout_error:#}, trying create_branch"
                );
                let create_result = wt_repo
                    .update(cx, |repo, _cx| {
                        repo.create_branch(branch_name.clone(), None)
                    })
                    .await;

                if let Err(error) = create_result.map_err(|e| anyhow!("{e}")).flatten() {
                    log::warn!(
                        "Failed to create branch '{branch_name}': {error:#}; \
                         restored worktree will be in detached HEAD state."
                    );
                }
            }
        }
    }

    // Restore the staged/unstaged state from the WIP commit trees.
    // read-tree --reset -u applies the unstaged tree (including deletions)
    // to the working directory, then a bare read-tree sets the index to
    // the staged tree without touching the working directory.
    let restore_rx = wt_repo.update(cx, |repo, _cx| {
        repo.restore_archive_checkpoint(
            row.staged_commit_hash.clone(),
            row.unstaged_commit_hash.clone(),
        )
    });
    if let Err(error) = restore_rx
        .await
        .map_err(|_| anyhow!("restore_archive_checkpoint canceled"))
        .and_then(|r| r)
    {
        let error = error.context("failed to restore archive checkpoint");
        remove_new_worktree_on_error(&main_repo, worktree_path, cx).await;
        rollback_backup(
            app_state.fs.as_ref(),
            backup.as_ref(),
            worktree_path,
            &error,
        )
        .await;
        return Err(error);
    }

    // The restore succeeded; drop the backup directory so the temporary
    // sibling doesn't linger on disk.
    if let Some(backup) = backup {
        if let Err(error) = app_state
            .fs
            .remove_dir(
                &backup.dir,
                RemoveOptions {
                    recursive: true,
                    ignore_if_not_exists: true,
                },
            )
            .await
        {
            log::warn!(
                "failed to clean up backup directory '{}' after successful restore: {error:#}",
                backup.dir.display()
            );
        }
    }

    Ok(worktree_path.clone())
}

/// Pre-existing content at `worktree_path` that was moved aside before the
/// destructive parts of [`restore_worktree_via_git`] ran. If anything goes
/// wrong, [`rollback_backup`] uses this to put the user's content back.
struct Backup {
    /// The temporary sibling directory holding the moved content.
    dir: PathBuf,
    /// Where inside `dir` the original entry now lives.
    target: PathBuf,
}

/// Restores the user's pre-existing content from a backup created by
/// [`restore_worktree_via_git`] back to `worktree_path`, then deletes the
/// now-empty backup directory.
///
/// On any rollback failure we log loudly (including the original error and
/// the backup path) so the user can recover manually, and leave the backup
/// directory in place. The original `restore_worktree_via_git` error is the
/// user-visible cause and is returned by the caller; we never propagate the
/// rollback error.
async fn rollback_backup(
    fs: &dyn Fs,
    backup: Option<&Backup>,
    worktree_path: &Path,
    original_error: &anyhow::Error,
) {
    let Some(backup) = backup else {
        return;
    };
    if let Ok(Some(metadata)) = fs.metadata(worktree_path).await {
        let is_empty_dir = if metadata.is_dir {
            use futures::stream::StreamExt as _;
            match fs.read_dir(worktree_path).await {
                Ok(mut entries) => entries.next().await.is_none(),
                Err(_) => false,
            }
        } else {
            false
        };
        if is_empty_dir {
            if let Err(clear_error) = fs
                .remove_dir(
                    worktree_path,
                    RemoveOptions {
                        recursive: false,
                        ignore_if_not_exists: true,
                    },
                )
                .await
            {
                log::warn!(
                    "failed to clear empty '{}' before rollback rename: {clear_error:#}",
                    worktree_path.display()
                );
            }
        } else {
            log::error!(
                "cannot rollback: '{}' has unexpected content after restore failure; \
                 original error: {original_error:#}; \
                 user's pre-existing content remains at '{}' for manual recovery",
                worktree_path.display(),
                backup.target.display(),
            );
            return;
        }
    }
    if let Err(rollback_error) = fs
        .rename(
            &backup.target,
            worktree_path,
            RenameOptions {
                overwrite: false,
                ignore_if_exists: false,
                create_parents: false,
            },
        )
        .await
    {
        log::error!(
            "failed to restore backup '{}' to '{}' after restore error: {rollback_error:#}; \
             original restore error: {original_error:#}; \
             user content remains at '{}' for manual recovery",
            backup.target.display(),
            worktree_path.display(),
            backup.target.display(),
        );
        return;
    }
    if let Err(cleanup_error) = fs
        .remove_dir(
            &backup.dir,
            RemoveOptions {
                recursive: true,
                ignore_if_not_exists: true,
            },
        )
        .await
    {
        log::warn!(
            "failed to clean up empty backup directory '{}' after rollback: {cleanup_error:#}",
            backup.dir.display()
        );
    }
}

/// Returns whether restoring this archived worktree would clobber any
/// pre-existing content on disk at the worktree's path.
///
/// Callers must invoke this **before** [`restore_worktree_via_git`] and prompt
/// the user for confirmation if it returns `true`, since the restore will
/// otherwise destroy that content.
pub async fn restore_would_overwrite(
    row: &ArchivedGitWorktree,
    remote_connection: Option<&RemoteConnectionOptions>,
    cx: &mut AsyncApp,
) -> Result<bool> {
    if remote_connection.is_some() {
        anyhow::bail!("restoring archived worktrees on remote machines is not yet supported");
    }
    let app_state = current_app_state(cx).context("no app state available")?;
    worktree_path_has_content(app_state.fs.as_ref(), &row.worktree_path).await
}

/// Returns whether the worktree path has any content that a restore would
/// destroy. A path that doesn't exist or that is an empty directory has no
/// content; anything else (a non-empty directory, or a file at this path)
/// counts as content.
async fn worktree_path_has_content(fs: &dyn Fs, path: &Path) -> Result<bool> {
    use futures::stream::StreamExt;

    let Some(metadata) = fs.metadata(path).await? else {
        return Ok(false);
    };

    if metadata.is_symlink {
        return Ok(true);
    }

    if !metadata.is_dir {
        return Ok(true);
    }

    let mut entries = fs.read_dir(path).await?;
    Ok(entries.next().await.is_some())
}

async fn remove_new_worktree_on_error(
    main_repo: &Entity<Repository>,
    worktree_path: &PathBuf,
    cx: &mut AsyncApp,
) {
    let rx = main_repo.update(cx, |repo, _cx| {
        repo.remove_worktree(worktree_path.clone(), true)
    });
    rx.await.ok().and_then(|r| r.log_err());
}

/// Deletes the git ref and DB records for a single archived worktree.
/// Used when an archived worktree is no longer referenced by any thread.
pub async fn cleanup_archived_worktree_record(
    row: &ArchivedGitWorktree,
    remote_connection: Option<&RemoteConnectionOptions>,
    cx: &mut AsyncApp,
) {
    // Delete the git ref from the main repo
    if let Ok((main_repo, _temp_project)) =
        find_or_create_repository(&row.main_repo_path, remote_connection, cx).await
    {
        let ref_name = archived_worktree_ref_name(row.id);
        let rx = main_repo.update(cx, |repo, _cx| repo.delete_ref(ref_name));
        match rx.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => log::warn!("Failed to delete archive ref: {error}"),
            Err(_) => log::warn!("Archive ref deletion was canceled"),
        }
        // See note in `remove_root_after_worktree_removal`: this may be a
        // live or temporary project; dropping only matters in the temporary
        // case.
        drop(_temp_project);
    }

    // Delete the DB records
    let store = cx.update(|cx| ThreadMetadataStore::global(cx));
    store
        .read_with(cx, |store, cx| store.delete_archived_worktree(row.id, cx))
        .await
        .log_err();
}

/// Cleans up all archived worktree data associated with a thread being deleted.
///
/// This unlinks the thread from all its archived worktrees and, for any
/// archived worktree that is no longer referenced by any other thread,
/// deletes the git ref and DB records.
pub async fn cleanup_thread_archived_worktrees(thread_id: ThreadId, cx: &mut AsyncApp) {
    let store = cx.update(|cx| ThreadMetadataStore::global(cx));
    let remote_connection = store.read_with(cx, |store, _cx| {
        store
            .entry(thread_id)
            .and_then(|t| t.remote_connection.clone())
    });

    let archived_worktrees = store
        .read_with(cx, |store, cx| {
            store.get_archived_worktrees_for_thread(thread_id, cx)
        })
        .await;
    let archived_worktrees = match archived_worktrees {
        Ok(rows) => rows,
        Err(error) => {
            log::error!("Failed to fetch archived worktrees for thread {thread_id:?}: {error:#}");
            return;
        }
    };

    if archived_worktrees.is_empty() {
        return;
    }

    if let Err(error) = store
        .read_with(cx, |store, cx| {
            store.unlink_thread_from_all_archived_worktrees(thread_id, cx)
        })
        .await
    {
        log::error!("Failed to unlink thread {thread_id:?} from archived worktrees: {error:#}");
        return;
    }

    for row in &archived_worktrees {
        let still_referenced = store
            .read_with(cx, |store, cx| {
                store.is_archived_worktree_referenced(row.id, cx)
            })
            .await;
        match still_referenced {
            Ok(true) => {}
            Ok(false) => {
                cleanup_archived_worktree_record(row, remote_connection.as_ref(), cx).await;
            }
            Err(error) => {
                log::error!(
                    "Failed to check if archived worktree {} is still referenced: {error:#}",
                    row.id
                );
            }
        }
    }
}

/// Collects every `Workspace` entity across all open `MultiWorkspace` windows.
pub fn all_open_workspaces(cx: &App) -> Vec<Entity<Workspace>> {
    cx.windows()
        .into_iter()
        .filter_map(|window| window.downcast::<MultiWorkspace>())
        .flat_map(|multi_workspace| {
            multi_workspace
                .read(cx)
                .map(|multi_workspace| multi_workspace.workspaces().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        })
        .collect()
}

fn current_app_state(cx: &mut AsyncApp) -> Option<Arc<AppState>> {
    cx.update(|cx| {
        all_open_workspaces(cx)
            .into_iter()
            .next()
            .map(|workspace| workspace.read(cx).app_state().clone())
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use git::repository::Worktree as GitWorktree;
    use gpui::{BorrowAppContext, TestAppContext};
    use project::Project;
    use serde_json::json;
    use settings::SettingsStore;
    use workspace::MultiWorkspace;

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
    }

    #[gpui::test]
    async fn test_build_root_plan_returns_none_for_main_worktree(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));

        let project = Project::test(fs.clone(), [Path::new("/project")], cx).await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        // The main worktree should NOT produce a root plan.
        workspace.read_with(cx, |_workspace, cx| {
            let plan = build_root_plan(
                Path::new("/project"),
                None,
                std::slice::from_ref(&workspace),
                cx,
            );
            assert!(
                plan.is_none(),
                "build_root_plan should return None for a main worktree",
            );
        });
    }

    #[gpui::test]
    async fn test_build_root_plan_returns_some_for_linked_worktree(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));
        fs.insert_branches(Path::new("/project/.git"), &["main", "feature"]);

        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/worktrees/project/feature/project"),
                ref_name: Some("refs/heads/feature".into()),
                sha: "abc123".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [
                Path::new("/project"),
                Path::new("/worktrees/project/feature/project"),
            ],
            cx,
        )
        .await;
        project
            .update(cx, |project, cx| project.git_scans_complete(cx))
            .await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        workspace.read_with(cx, |_workspace, cx| {
            // The linked worktree SHOULD produce a root plan.
            let plan = build_root_plan(
                Path::new("/worktrees/project/feature/project"),
                None,
                std::slice::from_ref(&workspace),
                cx,
            );
            assert!(
                plan.is_some(),
                "build_root_plan should return Some for a linked worktree",
            );
            let plan = plan.unwrap();
            assert_eq!(
                plan.root_path,
                PathBuf::from("/worktrees/project/feature/project")
            );
            assert_eq!(plan.main_repo_path, PathBuf::from("/project"));

            // The main worktree should still return None.
            let main_plan = build_root_plan(
                Path::new("/project"),
                None,
                std::slice::from_ref(&workspace),
                cx,
            );
            assert!(
                main_plan.is_none(),
                "build_root_plan should return None for the main worktree \
                 even when a linked worktree exists",
            );
        });
    }

    #[gpui::test]
    async fn test_build_root_plan_returns_none_for_external_linked_worktree(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));
        fs.insert_branches(Path::new("/project/.git"), &["main", "feature"]);

        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/external-worktree"),
                ref_name: Some("refs/heads/feature".into()),
                sha: "abc123".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [Path::new("/project"), Path::new("/external-worktree")],
            cx,
        )
        .await;
        project
            .update(cx, |project, cx| project.git_scans_complete(cx))
            .await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        workspace.read_with(cx, |_workspace, cx| {
            let plan = build_root_plan(
                Path::new("/external-worktree"),
                None,
                std::slice::from_ref(&workspace),
                cx,
            );
            assert!(
                plan.is_none(),
                "build_root_plan should return None for a linked worktree \
                 outside the Zed-managed worktrees directory",
            );
        });
    }

    #[gpui::test]
    async fn test_build_root_plan_with_custom_worktree_directory(cx: &mut TestAppContext) {
        init_test(cx);

        // Override the worktree_directory setting to a non-default location.
        // With main repo at /project and setting "../custom-worktrees", the
        // resolved base is /custom-worktrees/project.
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |s| {
                    s.git.get_or_insert(Default::default()).worktree_directory =
                        Some("../custom-worktrees".to_string());
                });
            });
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));
        fs.insert_branches(Path::new("/project/.git"), &["main", "feature", "feature2"]);

        // Worktree inside the custom managed directory.
        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/custom-worktrees/project/feature/project"),
                ref_name: Some("refs/heads/feature".into()),
                sha: "abc123".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        // Worktree outside the custom managed directory (at the default
        // `../worktrees` location, which is not what the setting says).
        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/worktrees/project/feature2/project"),
                ref_name: Some("refs/heads/feature2".into()),
                sha: "def456".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [
                Path::new("/project"),
                Path::new("/custom-worktrees/project/feature/project"),
                Path::new("/worktrees/project/feature2/project"),
            ],
            cx,
        )
        .await;
        project
            .update(cx, |project, cx| project.git_scans_complete(cx))
            .await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        workspace.read_with(cx, |_workspace, cx| {
            // Worktree inside the custom managed directory SHOULD be archivable.
            let plan = build_root_plan(
                Path::new("/custom-worktrees/project/feature/project"),
                None,
                std::slice::from_ref(&workspace),
                cx,
            );
            assert!(
                plan.is_some(),
                "build_root_plan should return Some for a worktree inside \
                 the custom worktree_directory",
            );

            // Worktree at the default location SHOULD NOT be archivable
            // because the setting points elsewhere.
            let plan = build_root_plan(
                Path::new("/worktrees/project/feature2/project"),
                None,
                std::slice::from_ref(&workspace),
                cx,
            );
            assert!(
                plan.is_none(),
                "build_root_plan should return None for a worktree outside \
                 the custom worktree_directory, even if it would match the default",
            );
        });
    }

    #[gpui::test]
    async fn test_remove_root_deletes_directory_and_git_metadata(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));
        fs.insert_branches(Path::new("/project/.git"), &["main", "feature"]);

        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/worktrees/project/feature/project"),
                ref_name: Some("refs/heads/feature".into()),
                sha: "abc123".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [
                Path::new("/project"),
                Path::new("/worktrees/project/feature/project"),
            ],
            cx,
        )
        .await;
        project
            .update(cx, |project, cx| project.git_scans_complete(cx))
            .await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        // Build the root plan while the worktree is still loaded.
        let root = workspace
            .read_with(cx, |_workspace, cx| {
                build_root_plan(
                    Path::new("/worktrees/project/feature/project"),
                    None,
                    std::slice::from_ref(&workspace),
                    cx,
                )
            })
            .expect("should produce a root plan for the linked worktree");

        assert!(
            fs.is_dir(Path::new("/worktrees/project/feature/project"))
                .await
        );

        // Remove the root.
        let task = cx.update(|cx| cx.spawn(async move |cx| remove_root(root, cx).await));
        task.await.expect("remove_root should succeed");

        cx.run_until_parked();

        // The FakeFs directory should be gone.
        assert!(
            !fs.is_dir(Path::new("/worktrees/project/feature/project"))
                .await,
            "linked worktree directory should be removed from FakeFs"
        );
    }

    #[gpui::test]
    async fn test_remove_root_succeeds_when_directory_already_gone(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));
        fs.insert_branches(Path::new("/project/.git"), &["main", "feature"]);

        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/worktrees/project/feature/project"),
                ref_name: Some("refs/heads/feature".into()),
                sha: "abc123".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [
                Path::new("/project"),
                Path::new("/worktrees/project/feature/project"),
            ],
            cx,
        )
        .await;
        project
            .update(cx, |project, cx| project.git_scans_complete(cx))
            .await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        let root = workspace
            .read_with(cx, |_workspace, cx| {
                build_root_plan(
                    Path::new("/worktrees/project/feature/project"),
                    None,
                    std::slice::from_ref(&workspace),
                    cx,
                )
            })
            .expect("should produce a root plan for the linked worktree");

        // Manually remove the worktree directory from FakeFs before calling
        // remove_root, simulating the directory being deleted externally.
        fs.as_ref()
            .remove_dir(
                Path::new("/worktrees/project/feature/project"),
                fs::RemoveOptions {
                    recursive: true,
                    ignore_if_not_exists: false,
                },
            )
            .await
            .unwrap();
        assert!(
            !fs.as_ref()
                .is_dir(Path::new("/worktrees/project/feature/project"))
                .await
        );

        // remove_root should still succeed — fs.remove_dir with
        // ignore_if_not_exists handles NotFound, and git worktree remove
        // handles a missing working tree directory.
        let task = cx.update(|cx| cx.spawn(async move |cx| remove_root(root, cx).await));
        task.await
            .expect("remove_root should succeed even when directory is already gone");
    }

    #[gpui::test]
    async fn test_remove_root_returns_error_and_rolls_back_on_remove_dir_failure(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/project",
            json!({
                ".git": {},
                "src": { "main.rs": "fn main() {}" }
            }),
        )
        .await;
        fs.set_branch_name(Path::new("/project/.git"), Some("main"));
        fs.insert_branches(Path::new("/project/.git"), &["main", "feature"]);

        fs.add_linked_worktree_for_repo(
            Path::new("/project/.git"),
            true,
            GitWorktree {
                path: PathBuf::from("/worktrees/project/feature/project"),
                ref_name: Some("refs/heads/feature".into()),
                sha: "abc123".into(),
                is_main: false,
                is_bare: false,
            },
        )
        .await;

        let project = Project::test(
            fs.clone(),
            [
                Path::new("/project"),
                Path::new("/worktrees/project/feature/project"),
            ],
            cx,
        )
        .await;
        project
            .update(cx, |project, cx| project.git_scans_complete(cx))
            .await;

        let multi_workspace =
            cx.add_window(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace
            .read_with(cx, |mw, _cx| mw.workspace().clone())
            .unwrap();

        cx.run_until_parked();

        let root = workspace
            .read_with(cx, |_workspace, cx| {
                build_root_plan(
                    Path::new("/worktrees/project/feature/project"),
                    None,
                    std::slice::from_ref(&workspace),
                    cx,
                )
            })
            .expect("should produce a root plan for the linked worktree");

        // Replace the worktree directory with a file so that fs.remove_dir
        // fails with a "not a directory" error.
        let worktree_path = Path::new("/worktrees/project/feature/project");
        fs.remove_dir(
            worktree_path,
            fs::RemoveOptions {
                recursive: true,
                ignore_if_not_exists: false,
            },
        )
        .await
        .unwrap();
        fs.create_file(worktree_path, fs::CreateOptions::default())
            .await
            .unwrap();
        assert!(
            fs.is_file(worktree_path).await,
            "path should now be a file, not a directory"
        );

        let task = cx.update(|cx| cx.spawn(async move |cx| remove_root(root, cx).await));
        let result = task.await;

        assert!(
            result.is_err(),
            "remove_root should return an error when fs.remove_dir fails"
        );
        let error_message = format!("{:#}", result.unwrap_err());
        assert!(
            error_message.contains("failed to delete worktree directory"),
            "error should mention the directory deletion failure, got: {error_message}"
        );

        cx.run_until_parked();

        // After rollback, the worktree should be re-added to the project.
        let has_worktree = project.read_with(cx, |project, cx| {
            project
                .worktrees(cx)
                .any(|wt| wt.read(cx).abs_path().as_ref() == worktree_path)
        });
        assert!(
            has_worktree,
            "rollback should have re-added the worktree to the project"
        );
    }

    /// Case B (the original bug): the worktree directory exists with leftover
    /// content (e.g. files that Windows couldn't fully delete during archival
    /// because they were locked by another process), but git has no
    /// registration for it. The pre-flight check must report content so the
    /// caller can prompt before the leftover files get clobbered.
    #[gpui::test]
    async fn test_has_content_leftover_dir_no_registration(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());

        // Worktree directory has leftover content but no .git linkage and no
        // entry in any main repo's .git/worktrees/.
        fs.insert_tree(
            "/wt-orphaned",
            json!({
                "leftover.txt": "important user data",
            }),
        )
        .await;

        let has_content = worktree_path_has_content(fs.as_ref(), Path::new("/wt-orphaned")).await;

        assert_eq!(
            has_content.unwrap(),
            true,
            "orphaned dir from a partial Windows archive must report content"
        );
        assert!(
            fs.is_file(Path::new("/wt-orphaned/leftover.txt")).await,
            "the check must not touch any files"
        );
    }

    /// Case D: the worktree directory exists with content but the `.git` file
    /// in the worktree itself is missing. The restore would still clobber
    /// any user files on disk, so we must report content here.
    #[gpui::test]
    async fn test_has_content_dir_with_broken_dot_git(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());

        // /wt-broken has files but no `.git` file — the linkage is broken.
        fs.insert_tree(
            "/wt-broken",
            json!({
                "src": { "lib.rs": "// existing user file" },
            }),
        )
        .await;

        let has_content = worktree_path_has_content(fs.as_ref(), Path::new("/wt-broken")).await;

        assert_eq!(
            has_content.unwrap(),
            true,
            "a directory with broken git linkage but real files must report content"
        );
    }

    /// Case E: the worktree directory is fully valid — `.git` file points back
    /// to the main repo. Even so, the restore will overwrite any uncommitted
    /// work the user has in there via `git read-tree --reset -u`, so we must
    /// still report content.
    #[gpui::test]
    async fn test_has_content_fully_valid_worktree(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());

        fs.insert_tree(
            "/wt-valid",
            json!({
                ".git": "gitdir: /project/.git/worktrees/feature",
                "src": { "lib.rs": "// uncommitted local work" },
            }),
        )
        .await;

        let has_content = worktree_path_has_content(fs.as_ref(), Path::new("/wt-valid")).await;

        assert_eq!(
            has_content.unwrap(),
            true,
            "a valid worktree with uncommitted work must report content \
             (read-tree --reset -u would clobber it)"
        );
    }

    /// Case A: nothing exists at the worktree path. The check must report
    /// no content — there's nothing to lose.
    #[gpui::test]
    async fn test_has_content_when_path_missing(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());

        assert!(
            fs.metadata(Path::new("/wt-missing"))
                .await
                .unwrap()
                .is_none(),
            "precondition: worktree path must not exist"
        );

        let has_content = worktree_path_has_content(fs.as_ref(), Path::new("/wt-missing")).await;

        assert_eq!(
            has_content.unwrap(),
            false,
            "missing dir must not report content"
        );
    }

    /// An empty directory at the worktree path has no content to lose.
    #[gpui::test]
    async fn test_has_content_for_empty_dir(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());

        fs.create_dir(Path::new("/wt-empty")).await.unwrap();
        assert!(
            fs.is_dir(Path::new("/wt-empty")).await,
            "precondition: empty dir must exist"
        );

        let has_content = worktree_path_has_content(fs.as_ref(), Path::new("/wt-empty")).await;

        assert_eq!(
            has_content.unwrap(),
            false,
            "empty dir must not report content"
        );
    }
}
