Vue.component("py-message-box", {
  props: ["id", "data"],
  template: `
<sly-message-box
    ref="message-box"
    :data="data"
/>
  `,
  methods: {
    showMessage(payload) {
      if (!payload?.data) return;

      const { data } = payload;

      if (!data) return;

      this.data = data;
      this.$refs["message-box"].open();
    },
  },

  created() {
    this.$eventBus.$on(`message-box-${this.id}`, this.showMessage);
  },
  beforeDestroy() {
    this.$eventBus.$off(`message-box-${this.id}`, this.showMessage);
  },
});
