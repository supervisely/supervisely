Vue.component("py-sly-video", {
  props: ["time_to_set", "video_url", "video_type"],
  template: `
<div>
    <video ref="video"
        width="100%"
        height="auto"
        controls
        @timeupdate="$emit('timeupdate', $refs['video'].currentTime)"
    >
        <source ref="video-data" :src="video_url" :type="video_type">
    </video>
</div>
`,

  watch: {
    time_to_set(time) {
      if (Number.isFinite(time)) {
        this.$refs["video"].currentTime = time;
        this.$emit("update:time_to_set", null);
      }
    },
    video_url: {
      handler() {
        this.update_video_src();
      },
      immediate: true,
    },
    video_type: {
      handler() {
        this.update_video_src();
      },
      immediate: true,
    },
  },
  methods: {
    update_video_src() {
      if (!this.video_url || !this.video_type) {
        return;
      }
      this.$refs["video"].pause();

      this.$refs["video-data"].setAttribute("src", this.video_url);
      this.$refs["video-data"].setAttribute("type", this.video_type);

      this.$refs["video"].load();
    },
  },
});
