Vue.component("sly-card2", {
  props: {
    title: {
      type: String,
      default: "",
    },
    description: {
      type: String,
      default: "",
    },
    properties: {
      type: Object,
      default: () => ({}),
    },
    propertiesLayout: {
      type: String,
      default: "vertical", // 'vertical' or 'horizontal'
      validator: (value) => ["vertical", "horizontal"].includes(value),
    },
    collapsable: {
      type: Boolean,
      default: false,
    },
    removePadding: {
      type: Boolean,
      default: false,
    },
    overflow: {
      type: String,
      default: "auto",
    },
    lockMessage: {
      type: String,
      default: "Card content is locked",
    },
    isLocked: {
      type: Boolean,
      default: false,
    },
    hasIcon: {
      type: Boolean,
      default: false,
    },
    hasContent: {
      type: Boolean,
      default: false,
    },
    hasContentTopRight: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      collapsed: false,
    };
  },
  computed: {
    hasProperties() {
      return this.properties && Object.keys(this.properties).length > 0;
    },
    propertiesArray() {
      // properties - object with key-value pairs (value is an array of arrays [value, type], where type is one of the following: 'text', 'tag')
      // Convert properties object to array for easier rendering
      return Object.entries(this.properties).map(([key, value]) => ({
        key,
        value: Array.isArray(value) ? value : [value],
      }));
    },
    hasHeader() {
      return (
        this.title ||
        this.description ||
        this.hasIcon ||
        this.hasContentTopRight ||
        this.collapsable
      );
    },
  },
  watch: {
    properties: {
      handler(newValue) {
        this.propertiesArray = Object.entries(newValue).map(([key, value]) => ({
          key,
          value: Array.isArray(value) ? value : [value],
        }));
      },
      immediate: true,
    },
    mounted() {
      this.$emit("update:properties", this.propertiesArray);
    },
  },
  methods: {
    toggleCollapse() {
      this.collapsed = !this.collapsed;
    },
  },
  template: `
    <div class="sly-card2" :class="{'no-padding': removePadding}">
      <!-- Header -->
      <div class="sly-card2-header" v-if="hasHeader">
        <!-- Icon slot -->
        <div v-if="hasIcon" class="sly-card2-header-icon">
          <slot name="icon"></slot>
        </div>

        <div class="sly-card2-header-content">
          <h3 v-if="title" class="sly-card2-title">{{ title }}</h3>
          <div v-if="description" class="sly-card2-description">{{ description }}</div>
        </div>

        <!-- Top right content slot -->
        <div v-if="hasContentTopRight" class="sly-card2-header-slot">
          <slot name="content-top-right"></slot>
        </div>

        <!-- Collapse button -->
        <div v-if="collapsable" class="sly-card2-header-actions">
          <span @click="toggleCollapse">
            <i :class="collapsed ? 'zmdi zmdi-chevron-down' : 'zmdi zmdi-chevron-up'"></i>
          </span>
        </div>
      </div>

    <div :class="[collapsed ? 'collapsed-content' : '']" v-if="hasContent || hasProperties">
      <!-- Divider -->
      <div class="sly-card2-divider" v-if="hasProperties && hasHeader"></div>

      <template v-if="hasProperties" >
        <div class="sly-card2-properties" :class="[
          propertiesLayout === 'vertical' ? 'sly-card2-properties-vertical' : 'sly-card2-properties-horizontal'
        ]">
          <!-- For vertical layout -->
          <template v-if="propertiesLayout === 'vertical'">
            <div class="sly-card2-property-item" v-for="prop in propertiesArray" :key="prop.key">
              <span class="sly-card2-property-name">{{ prop.key }}:</span>
              <!-- Display each value in the array -->
              <div class="sly-card2-property-value">
                <template v-for="(item, index) in prop.value">
                  <span :class="[item[1] === 'tag' ? 'sly-card2-property-value-item sly-card2-property-value-tag' : 'sly-card2-property-value-item']" :key="index">{{ item[0] }}</span>
                </template>
              </div>

            </div>
          </template>

          <!-- For horizontal layout -->
          <template v-else>
            <div class="sly-card2-properties-content">
              <template v-for="(prop, index) in propertiesArray">
                <!-- Separator -->
                <span v-if="index > 0" class="sly-card2-properties-separator" :key="'sep-'+index">•</span>
                <!-- Display each value in the array -->
                <template v-for="(item, index) in prop.value">
                  <span :class="[item[1] === 'tag' ? 'sly-card2-property-value-item sly-card2-property-value-tag' : 'sly-card2-property-value-item']" :key="index">{{ item[0] }}</span>
                </template>
              
              </template>
            </div>
          </template>
        </div>
        </template>
        
        <!-- Content -->
        <div class="sly-card2-divider" v-if="hasContent && (hasProperties || hasHeader)"></div>
        <div class="sly-card2-content" v-if="hasContent"  
            
            :style="{ overflow: overflow }">
            <slot></slot>
        </div>
      </div>
    </div>
  `,
});
