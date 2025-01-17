Vue.component('sly-run-app-button', {
  props: {
    'publicApiInstance': { 'type': Object },
    'workspaceId': { 'type': Number }, 'moduleId': { 'type': Number }, 'payload': { 'type': Object }, 'options': { 'type': Object }
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
  
          const tasks = await this.publicApiInstance.post('tasks.run.app', {
            payload: this.payload,
            workspaceId: this.workspaceId,
            moduleId: this.moduleId,
            nodeId: null,
          }).then(response => response.data);
  
          const task = tasks[0];
          console.log('> Base URL:', this.publicApiInstance.defaults.baseURL);
          const origin = new URL(this.publicApiInstance.defaults.baseURL).origin;
          console.log('> ORIGIN:', origin);
          console.log('> TASK:', task);
          window.open(`${origin}/apps/${task.appId}/sessions/${task.taskId}`, '_blank');
        } finally {
          this.loading = false;
        }
      }
    },
    template: `
  <el-button
    :class="{'available-in-offline': options.available_in_offline}"
    @click="runApp"
    v-loading="options.loading || loading"
    :type="options.button_type"
    :plain="options.plain"
    :size="options.size"
    :disabled="options.disabled"
    
  >
  <span v-html="options.icon"></span>
  <span v-html="options.text"></span>
  </el-button>
  `,
});
