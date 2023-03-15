Vue.component("html-video", {
  props: ["timeToSet", "url", "mimeType", "isPlaying", "maskPath"],
  template: `
<div style="position: relative;">
    <video ref="video"
        width="100%"
        height="auto"
        controls
        @timeupdate="timeUpdated"
        @play="$emit('update:is-playing', true)"
        @pause="$emit('update:is-playing', false)"
        preload="metadata"
    >
        <source ref="video-data" :src="url" :type="mimeType">
    </video>
    <div v-if="maskPath"
        ref="mask"
        style="
            opacity: 0.4;
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            background-size: contain;
        " 
        :style="overlayStyle">
    </div>
</div>
`,
  computed: {
    overlayStyle() {
      return {
        backgroundImage: `url("${this.maskPath}")`,
      };
    },
  },
  watch: {
    timeToSet(time) {
      console.log("timeToSet: ", time, Number.isFinite(time));
      if (Number.isFinite(time)) {
        this.$refs["video"].currentTime = time;
        this.$emit("update:time-to-set", null);
      }
    },
    url: {
      handler() {
        console.log("URL: ", this.url);
        this.updateVideoSrc();
      },
    },
    mimeType: {
      handler() {
        console.log("mimeType: ", this.mimeType);
        this.updateVideoSrc();
      },
    },
    isPlaying: {
      handler(value) {
        this.playPause(value);
      },
    },
  },
  mounted() {
    console.log("mounted", this.$refs["video"]);
    this.updateVideoSrc();
    if (this.isPlaying) {
      this.$emit("update:is-playing", false);
    }
  },
  methods: {
    updateVideoSrc() {
      const video = this.$refs["video"];
      const source = this.$refs["video-data"];
      console.log("updateVideoSrc video", video);
      console.log("updateVideoSrc source", source);
      console.log("updateVideoSrc url", this.url);
      console.log("updateVideoSrc mimeType", this.mimeType);
      if (!this.url || !this.mimeType || !video) {
        return;
      }
      video.pause();
      source.setAttribute("src", this.url);
      source.setAttribute("type", this.mimeType);
      //   video.src = this.url;
      //   video.setAttribute("src", this.url);
      //    <source ref="video-data" src="" type=""></source>
      console.log("updateVideoSrc: SUCCESS");
      this.$nextTick(() => {
        video.load();
      });
    },
    playPause() {
      const video = this.$refs["video"];
      if (!video) {
        return;
      }
      this.isPlaying ? video.play() : video.pause();
    },
    timeUpdated() {
      console.log("ref video:", this.$refs["video"]);
      console.log("currentTime:", this.$refs["video"].currentTime);
      this.$emit("timeupdate", this.$refs["video"].currentTime);
    },
  },
});
