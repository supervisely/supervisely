// Hotkeys Vue 2 component - catches keyboard shortcuts anywhere on the page.
Vue.component('sly-hotkeys', {
  template: '<span style="display:none"></span>',

  props: {
    hotkeys:          { type: Array,   default: function() { return []; } },
    preventDefault:   { type: Boolean, default: true },
    ignoreInputFocus:  { type: Boolean, default: true },
  },

  methods: {
    // Builds a normalized "ctrl+shift+s"-like combo string from a KeyboardEvent.
    // Modifier order is always ctrl, alt, shift to match how combos are declared in Python.
    normalizeCombo: function(e) {
      var parts = [];
      if (e.ctrlKey || e.metaKey) parts.push('ctrl');
      if (e.altKey) parts.push('alt');
      if (e.shiftKey) parts.push('shift');

      var key = e.key === ' ' ? 'space' : String(e.key).toLowerCase();
      if (['control', 'alt', 'shift', 'meta'].indexOf(key) === -1) {
        parts.push(key);
      }
      return parts.join('+');
    },

    // Legitimate user input (typing in a field) always takes priority over hotkeys.
    isTypingTarget: function(el) {
      if (!el) return false;
      var tag = el.tagName ? el.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
      if (el.isContentEditable) return true;
      return false;
    },

    handleKeydown: function(e) {
      if (this.ignoreInputFocus && this.isTypingTarget(document.activeElement)) return;

      var combo = this.normalizeCombo(e);
      if (this.hotkeys.indexOf(combo) !== -1) {
        if (this.preventDefault) e.preventDefault();
        this.$emit('hotkey-pressed', combo);
      }
    },
  },

  mounted: function() {
    this._onKeydown = this.handleKeydown.bind(this);
    document.addEventListener('keydown', this._onKeydown);
  },

  beforeDestroy: function() {
    document.removeEventListener('keydown', this._onKeydown);
  },
});
