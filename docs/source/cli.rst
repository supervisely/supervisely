Supervisely command-line interface
==================================

The ``supervisely`` executable provides operational commands for teams,
workspaces, projects, datasets, agents, Ecosystem applications, tasks, and Team
Files. It uses ``SERVER_ADDRESS`` and ``API_TOKEN`` from the normal SDK
environment (including ``~/supervisely.env`` in development). Credentials can
be overridden for one invocation with the global ``--server-address`` and
``--api-token`` options.

Use the global ``--json`` option before the command group when output will be
consumed by another program::

   supervisely --json project list --workspace-id 8
   supervisely --json task status --id 12345

Without ``--json``, API-backed commands render compact human-readable tables.
Legacy project transfer, Team Files, release, and task-output commands remain
available.

Application discovery and parameters
------------------------------------

Applications are selected by Ecosystem slug. The CLI resolves the internal
module ID through the API::

   supervisely app list --search export
   supervisely --json app describe \
       --slug supervisely-ecosystem/export-to-supervisely-format

Before starting an app, ``app params`` reports the app's declared modal state,
defaults, and any required context-menu target. Its JSON response includes a
``params_template`` and the expected key and primitive type for context targets::

   supervisely --json app params \
       --slug supervisely-ecosystem/export-to-supervisely-format \
       | jq '.params_template' > params.json

Replace any context placeholder in ``params.json`` with the required project,
dataset, team, job, or Team Files value. Ecosystem configuration does not expose
a universal validation schema, so the helper cannot describe rules that an app
does not declare.

Run applications
----------------

Run the latest release, a selected release, or a repository branch::

   supervisely app run \
       --slug supervisely-ecosystem/export-to-supervisely-format \
       --workspace-id 8 \
       --params-file params.json

   supervisely app run \
       --slug owner/application \
       --workspace-id 8 \
       --version v1.2.0

   supervisely app run \
       --slug owner/application \
       --workspace-id 8 \
       --branch feature/new-configuration

``--branch`` and ``--version`` are mutually exclusive. ``--params-file`` may
contain any JSON object accepted by ``api.app.start(params=...)``; the CLI does
not discard unknown app-specific fields. Omit ``--agent-id`` to let Supervisely
select an eligible agent. If ``--params-file`` is omitted, the app's declared
modal defaults are used.

Stopping an app session or task requires an explicit ``--yes`` flag::

   supervisely app stop --task-id 12345 --yes
   supervisely task stop --id 12345 --yes
