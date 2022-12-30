Vue.component("html-video", {
  props: ["time_to_set", "url", "mime_type", "playing"],
  template: `
<div>
    <video ref="video"
        width="100%"
        height="auto"
        controls
        @timeupdate="$emit('timeupdate', $refs['video'].currentTime)"
    >
        <source ref="video-data" :src="url" :type="mime_type">
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
    url: {
      handler() {
        this.update_video_src();
      },
      immediate: true,
    },
    mime_type: {
      handler() {
        this.update_video_src();
      },
      immediate: true,
    },
    playing: {
      handler() {
        this.play_pause();
      },
      immediate: true,
    },
  },
  methods: {
    update_video_src() {
      if (!this.url || !this.mime_type || !this.$refs["video"]) {
        return;
      }
      this.$refs["video"].pause();

      this.$refs["video-data"].setAttribute("src", this.url);
      this.$refs["video-data"].setAttribute("type", this.mime_type);

      this.$refs["video"].load();
    },
    play_pause() {
      if (!this.$refs["video"]) {
        return;
      }
      if (this.$refs["video"] && this.playing === false) {
        this.$refs["video"].pause();
      } else {
        this.$refs["video"].play();
      }
    },
  },
});
