Vue.component('sly-py-labeled-image', {
  props: ['imgUrl', 'projectMeta', 'annotation', 'options', 'visibleClasses'],
  data() {
    return {
      imageWidgetState: null,
      imageWidgetStyles: {
        width: '100%',
        height: '300px',
      },
      imageInfo: null,
    };
  },
  methods: {
    imageLoaded(imageInfo) {
      this.imageInfo = imageInfo;
      this.resizeView();
    },
    resizeView() {
      if (!this.imageInfo) return;
      const style = getComputedStyle(this.$el);
      if (!style?.width) return;
      const maxWidth = Math.round(parseFloat(style.width));
      const ratio = maxWidth / this.imageInfo.width;
      const height = `${Math.max(Math.floor(this.imageInfo.height * ratio), 100)}px`;
      this.imageWidgetStyles.height = height;
    }
  },
  template: `
<div class="sly-py-labeled-image" style="position: relative;">
  <div class="fflex">
    <el-button @click="imageWidgetState.fitImage()" type="text"><i class="zmdi zmdi-aspect-ratio-alt mr5"></i>Reset zoom</el-button>
    <el-slider
      :min="0"
      :max="1"
      :step="0.1"
      style="width: 150px; margin-left: 30px;"
      v-model="options.opacity"
    ></el-slider>
  </div>
  <sly-img
    @exposed="imageWidgetState = $event"
    :style="imageWidgetStyles"
    :image-url="imgUrl"
    :project-meta="projectMeta"
    :annotation="annotation"
    :options="options"
    @image-info="imageLoaded"
    @resize="resizeView"
  />
</div>
  `,
});