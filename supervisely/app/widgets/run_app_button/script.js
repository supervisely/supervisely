Vue.component("sly-run-app-button", {
	props: {
		publicApiInstance: { type: Function },
		workspaceId: { type: Number },
		moduleId: { type: Number },
		payload: { type: Object },
		options: { type: Object, default: () => ({}) },
		groupId: { type: Number },
		checkExistingTaskCb: { type: Function, default: null },
	},
	data() {
		return {
			loading: false,
		};
	},
	methods: {
		async runApp() {
			try {
				this.loading = true;

				if (this.checkExistingTaskCb) {
					let checkExistingTaskCb =
						typeof this.checkExistingTaskCb === "function"
							? this.checkExistingTaskCb
							: null;

					if (typeof this.checkExistingTaskCb === "string") {
						console.log("checkExistingTaskCb", this.checkExistingTaskCb);
						try {
							checkExistingTaskCb = new Function(
								"task",
								this.checkExistingTaskCb
							);
						} catch (err) {
							console.log("Error parsing checkExistingTaskCb string:", err);
						}
					}

					console.log("Before check checkExistingTaskCb", checkExistingTaskCb);
					if (checkExistingTaskCb) {
						const allEntities = await this.publicApiInstance
							.post("apps.list", {
								withShared: true,
								onlyRunning: true,
								groupId: this.groupId,
								filter: [
									{ field: "moduleId", operator: "=", value: this.moduleId },
								],
							})
							.then((res) => res.data?.entities || []);

						const existTasks = allEntities.flatMap(
							(entity) => entity.tasks || []
						);

						if (existTasks.length) {
							const foundTask = existTasks.find((t) => checkExistingTaskCb(t));
							console.log("foundTask", foundTask);

							if (foundTask) {
								window.open(
									`/apps/${foundTask.meta.app.id}/sessions/${foundTask.id}`,
									"_blank"
								);
								return;
							}
						}
					}
				}

				const tasks = await this.publicApiInstance
					.post("tasks.run.app", {
						params: this.payload,
						workspaceId: this.workspaceId,
						moduleId: this.moduleId,
						nodeId: null,
					})
					.then((response) => response.data);

				const task = tasks[0];
				const origin = new URL(this.publicApiInstance.defaults.baseURL).origin;
				window.open(
					`${origin}/apps/${task.appId}/sessions/${task.taskId}`,
					"_blank"
				);
			} finally {
				this.loading = false;
			}
		},
	},
	template: `
    <el-button
      :class="{'available-in-offline': options.available_in_offline}"
      @click="runApp"
      v-loading="options.loading || loading"
      :type="options.button_type"
      :plain="options.plain"
      :size="options.button_size"
      :disabled="options.disabled"
      
    >
    <span v-html="options.icon"></span>
    <span v-html="options.text"></span>
    </el-button>
    `,
});
