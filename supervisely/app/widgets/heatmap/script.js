Vue.component('heatmap-image', {
  template: `
  <div
    @click="$emit('click')"
  >
    <div class="heatmap-header" :style="widthStyle">
      <div class="opacity-slider" @click.stop>
        <el-slider
        type="range"
        :min="0"
        :max="100"
        :step="1"
        :value="opacity"
        @input="$emit('update:opacity', $event)"
        class="slider"
        >
        <span class="opacity-label">{{ opacity }}%</span>
      </div>
      
      <div class="legend" @click.stop>
        <span class="legend-label legend-min">{{ minValue }}</span>
        <div class="legend-gradient" :style="{ background: gradientStyle }"></div>
        <span class="legend-label legend-max">{{ maxValue }}</span>
      </div>
    </div>

    <div
      class="image-wrapper"
      :style="imageWrapperStyle"
    >
      <img
      class="base-image"
      :src="backgroundUrl"
      @load="handleImageLoad"
      >
      <img
      class="overlay-image"
      :style="{ opacity: opacity / 100 }"
      :src="maskUrl"
      >
    </div>
  </div>
  `,
  data() {
    return {
      naturalWidth: null,
      naturalHeight: null
    };
  },
  computed: {
    gradientStyle() {
      return `linear-gradient(to right, ${this.legendColors.join(', ')})`;
    },
    widthStyle() {
      const styles = {};
      
      if (this.width) {
        styles.width = typeof this.width === 'number' ? `${this.width}px` : this.width;
      } else if (this.height && this.naturalWidth && this.naturalHeight) {
        const heightValue = typeof this.height === 'number' ? this.height : parseFloat(this.height);
        const aspectRatio = this.naturalWidth / this.naturalHeight;
        styles.width = `${heightValue * aspectRatio}px`;
      }
      
      return styles;
    },
    imageWrapperStyle() {
      const styles = { ...this.widthStyle };
      
      if (this.naturalWidth && this.naturalHeight) {
        styles.aspectRatio = `${this.naturalWidth} / ${this.naturalHeight}`;
      }
      
      return styles;
    }
  },
  methods: {
    handleImageLoad(event) {
      this.naturalWidth = event.target.naturalWidth;
      this.naturalHeight = event.target.naturalHeight;
    }
  },
  watch: {
    backgroundUrl() {
      this.naturalWidth = null;
      this.naturalHeight = null;
    }
  },
  props: {
    backgroundUrl: String,
    maskUrl: String,
    opacity: Number,
    width: [Number, String],
    height: [Number, String],
    legendColors: {
      type: Array,
      default: () => ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    },
    minValue: Number,
    maxValue: Number,
  },
});