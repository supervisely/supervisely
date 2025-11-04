Vue.component('heatmap-image', {
  template: `
  <div
    @click="$emit('click')"
  >
    <div class="heatmap-header">
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
      :style="{ aspectRatio: \`\${ width } / \${ height }\` }"
    >
      <img
      class="base-image"
      :src="backgroundUrl"
      >
      <img
      class="overlay-image"
      :style="{ opacity: opacity / 100 }"
      :src="maskUrl"
      >
    </div>
  </div>
  `,
  computed: {
    gradientStyle() {
      return `linear-gradient(to right, ${this.legendColors.join(', ')})`;
    }
  },
  props: {
    backgroundUrl: String,
    maskUrl: String,
    opacity: Number,
    width: Number,
    height: Number,
    legendColors: {
      type: Array,
      default: () => ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    },
    minValue: Number,
    maxValue: Number,
  },
});
