// ReorderTable Vue 2 component
Vue.component('reorder-table', {
  template:
    '<div class="sly-reorder-wrap">' +

      // ── Main table panel ─────────────────────────────────────────────
      '<div class="sly-reorder-table">' +

        // Search bar
        '<div class="sly-rt-search-bar">' +
          '<el-input' +
          '  v-model="searchQuery"' +
          '  placeholder="Search..."' +
          '  prefix-icon="el-icon-search"' +
          '  clearable' +
          '  size="small"' +
          '/>' +
        '</div>' +

        '<div class="sly-rt-wrapper">' +
          '<table class="sly-rt-table">' +

            // Header row
            '<thead>' +
              '<tr>' +
                '<th class="sly-rt-th sly-rt-cb-col">' +
                  '<el-checkbox' +
                  '  :key="\'hcb-\' + (pageAllSelected ? 1 : 0) + \'-\' + (pageSomeSelected ? 1 : 0)"' +
                  '  :checked="pageAllSelected"' +
                  '  :indeterminate="pageSomeSelected && !pageAllSelected"' +
                  '  @change="handleHeaderCheckbox"' +
                  '  style="transform:scale(1.15)"' +
                  '/>' +
                '</th>' +
                '<th class="sly-rt-th sly-rt-drag-col"></th>' +
                '<th class="sly-rt-th sly-rt-order-col">#</th>' +
                '<th v-for="col in columns" :key="col" class="sly-rt-th sly-rt-data-col">{{ col }}</th>' +
              '</tr>' +
            '</thead>' +

            // Body rows
            '<tbody>' +
              '<tr' +
              '  v-for="item in pageItems"' +
              '  :key="item.globalPos"' +
              '  :class="[\'sly-rt-row\',' +
              '    selectedArr.indexOf(item.globalPos) !== -1 ? \'sly-rt-row-selected\' : \'\',' +
              '    dragOverPos === item.globalPos ? \'sly-rt-row-dragover\' : \'\'' +
              '  ]"' +
              '  @dragover.prevent="onDragOver($event, item)"' +
              '  @dragleave="onDragLeave($event)"' +
              '  @drop.prevent="onDrop($event, item)"' +
              '  @dragend="onDragEnd"' +
              '>' +
                // Checkbox cell
                '<td class="sly-rt-td sly-rt-cb-col" @click.stop>' +
                  '<el-checkbox' +
                  '  v-model="selectedArr"' +
                  '  :label="item.globalPos"' +
                  '  @change="$emit(\'update:selectedPositions\', selectedArr.slice())"' +
                  '  style="transform:scale(1.15)"' +
                  '/>' +
                '</td>' +

                // Drag handle cell — only this element is draggable
                '<td class="sly-rt-td sly-rt-drag-col">' +
                  '<span' +
                  '  draggable="true"' +
                  '  class="sly-rt-drag-handle"' +
                  '  title="Drag to reorder"' +
                  '  @dragstart="onDragStart($event, item)"' +
                  '>' +
                    '<i class="zmdi zmdi-menu"></i>' +
                  '</span>' +
                '</td>' +

                // Order badge cell
                '<td class="sly-rt-td sly-rt-order-col">' +
                  '<span class="sly-rt-orig-badge">{{ item.origPos }}</span>' +
                  '<span v-if="item.origPos !== item.currentPos" class="sly-rt-new-badge">' +
                    '&#8594; {{ item.currentPos }}' +
                  '</span>' +
                '</td>' +

                // Data cells
                '<td v-for="(cell, ci) in item.row" :key="ci" class="sly-rt-td sly-rt-data-col">' +
                  '<input type="text" readonly tabindex="-1" :value="String(cell)" class="sly-rt-cell-txt" />' +
                '</td>' +
              '</tr>' +

              // Empty state
              '<tr v-if="pageItems.length === 0">' +
                '<td :colspan="(columns ? columns.length : 0) + 3" class="sly-rt-empty">No items</td>' +
              '</tr>' +
            '</tbody>' +

          '</table>' +
        '</div>' + // sly-rt-wrapper

        // ── Pagination footer ─────────────────────────────────────────
        '<div class="sly-rt-footer">' +

          // Page-size control
          '<div class="sly-rt-page-size-ctrl">' +
            '<span class="sly-rt-footer-label">Rows per page:&nbsp;</span>' +
            '<span v-if="!editingPageSize" class="sly-rt-pagesize-val" @click="startEditPageSize">{{ localPageSize }}</span>' +
            '<input' +
            '  v-else' +
            '  ref="pageSizeInput"' +
            '  v-model.number="pageSizeEdit"' +
            '  type="number"' +
            '  min="1"' +
            '  class="sly-rt-pagesize-input"' +
            '  @keydown.enter="confirmPageSize"' +
            '  @keydown.esc="cancelEditPageSize"' +
            '  @blur="confirmPageSize"' +
            '/>' +
          '</div>' +

          // Page navigation
          '<div class="sly-rt-pagination">' +
            '<span class="sly-rt-range-label">{{ rangeLabel }}</span>' +
            '<button class="sly-rt-pg-btn" :disabled="localPage <= 1" @click="goPage(1)" title="First page">' +
              '<i class="zmdi zmdi-skip-previous"></i>' +
            '</button>' +
            '<button class="sly-rt-pg-btn" :disabled="localPage <= 1" @click="goPage(localPage - 1)" title="Previous page">' +
              '<i class="zmdi zmdi-chevron-left"></i>' +
            '</button>' +
            '<span class="sly-rt-page-label">{{ localPage }} / {{ totalPages }}</span>' +
            '<button class="sly-rt-pg-btn" :disabled="localPage >= totalPages" @click="goPage(localPage + 1)" title="Next page">' +
              '<i class="zmdi zmdi-chevron-right"></i>' +
            '</button>' +
            '<button class="sly-rt-pg-btn" :disabled="localPage >= totalPages" @click="goPage(totalPages)" title="Last page">' +
              '<i class="zmdi zmdi-skip-next"></i>' +
            '</button>' +
          '</div>' +

        '</div>' + // sly-rt-footer

      '</div>' + // sly-reorder-table

      // ── Floating action panel (shown when ≥1 row selected) ────────
      '<transition name="sly-rt-slide">' +
        '<div v-if="selectedArr.length > 0" class="sly-rt-action-panel">' +

          '<div class="sly-rt-ap-title">{{ selectedArr.length }} item{{ selectedArr.length !== 1 ? \'s\' : \'\' }} selected</div>' +

          '<button class="sly-rt-ap-btn" @click="moveToTop" title="Move selection to top">' +
            '<i class="zmdi zmdi-long-arrow-up"></i> Top' +
          '</button>' +

          '<button class="sly-rt-ap-btn" @click="moveUp" title="Move selection up one position">' +
            '<i class="zmdi zmdi-chevron-up"></i> Up' +
          '</button>' +

          // Set-to-position control
          '<div class="sly-rt-ap-setto">' +
            '<button v-if="!showSetTo" class="sly-rt-ap-btn sly-rt-ap-btn-secondary" @click="openSetTo">' +
              '<i class="zmdi zmdi-swap-vertical"></i> Set to #' +
            '</button>' +
            '<div v-else class="sly-rt-ap-setto-form">' +
              '<input' +
              '  ref="setToInput"' +
              '  v-model.number="setToValue"' +
              '  type="number"' +
              '  min="1"' +
              '  :max="localOrder.length"' +
              '  class="sly-rt-setto-input"' +
              '  placeholder="pos"' +
              '  @keydown.enter="applySetTo"' +
              '  @keydown.esc="showSetTo = false"' +
              '/>' +
              '<button class="sly-rt-ap-btn-confirm" @click="applySetTo" title="Confirm">&#10003;</button>' +
              '<button class="sly-rt-ap-btn-cancel" @click="showSetTo = false" title="Cancel">&#10007;</button>' +
            '</div>' +
          '</div>' +

          '<button class="sly-rt-ap-btn" @click="moveDown" title="Move selection down one position">' +
            '<i class="zmdi zmdi-chevron-down"></i> Down' +
          '</button>' +

          '<button class="sly-rt-ap-btn" @click="moveToBottom" title="Move selection to bottom">' +
            '<i class="zmdi zmdi-long-arrow-down"></i> Bottom' +
          '</button>' +

          '<div class="sly-rt-ap-divider"></div>' +

          '<button class="sly-rt-ap-btn sly-rt-ap-btn-deselect" @click="clearSelection">' +
            '<i class="zmdi zmdi-close"></i> Deselect' +
          '</button>' +

        '</div>' +
      '</transition>' +

    '</div>', // sly-reorder-wrap

  // ── Props ────────────────────────────────────────────────────────────
  props: {
    widgetId:          { type: String, default: '' },
    columns:           { type: Array,  default: function() { return []; } },
    rows:              { type: Array,  default: function() { return []; } },
    total:             { type: Number, default: 0 },
    order:             { type: Array,  default: function() { return []; } },
    page:              { type: Number, default: 1 },
    pageSize:          { type: Number, default: 10 },
    selectedPositions: { type: Array,  default: function() { return []; } },
  },

  // ── Local data ───────────────────────────────────────────────────────
  data: function() {
    return {
      localOrder:       this.order             ? this.order.slice()             : [],
      localPage:        this.page              || 1,
      localPageSize:    this.pageSize          || 10,
      selectedArr:      this.selectedPositions ? this.selectedPositions.slice() : [],
      searchQuery:      '',
      debouncedQuery:   '',
      searchTimer:      null,
      dragOverPos:      null,
      isDragging:       false,
      dragItem:         null,
      showSetTo:        false,
      setToValue:       1,
      editingPageSize:  false,
      pageSizeEdit:     this.pageSize || 10,
    };
  },

  // ── Computed ─────────────────────────────────────────────────────────
  computed: {
    // Positions (indices into localOrder) that match the current search query.
    // When no query is set, all positions are included.
    filteredOrder: function() {
      var self = this;
      if (!self.debouncedQuery || !self.debouncedQuery.trim()) {
        return self.localOrder.map(function(_, i) { return i; });
      }
      var q = self.debouncedQuery.trim().toLowerCase();
      return self.localOrder.reduce(function(acc, origIdx, localPos) {
        var row = self.rows[origIdx] || [];
        var matches = row.some(function(cell) {
          return String(cell).toLowerCase().indexOf(q) !== -1;
        });
        if (matches) acc.push(localPos);
        return acc;
      }, []);
    },
    totalPages: function() {
      return Math.max(1, Math.ceil(this.filteredOrder.length / this.localPageSize));
    },
    pageStart: function() {
      return (this.localPage - 1) * this.localPageSize;
    },
    pageEnd: function() {
      return Math.min(this.pageStart + this.localPageSize, this.filteredOrder.length);
    },
    rangeLabel: function() {
      if (this.filteredOrder.length === 0) {
        return this.debouncedQuery ? '0 results' : '0 items';
      }
      return (this.pageStart + 1) + '\u2013' + this.pageEnd + ' of ' + this.filteredOrder.length;
    },
    pageItems: function() {
      var self = this;
      if (!self.rows || !self.localOrder) return [];
      return self.filteredOrder.slice(self.pageStart, self.pageEnd).map(function(localPos, pagePos) {
        var origIdx = self.localOrder[localPos];
        return {
          origIdx:    origIdx,
          origPos:    origIdx + 1,
          currentPos: localPos + 1,
          globalPos:  localPos,
          row:        self.rows[origIdx] || [],
          pagePos:    pagePos,
        };
      });
    },
    // Two-way computed for the header checkbox (select-all on current page)
    pageAllSelected: {
      get: function() {
        var self = this;
        return this.pageItems.length > 0 && this.pageItems.every(function(item) {
          return self.selectedArr.indexOf(item.globalPos) !== -1;
        });
      },
      set: function(val) {
        this.toggleAll(val);
      },
    },
    pageSomeSelected: function() {
      var self = this;
      return this.pageItems.some(function(item) {
        return self.selectedArr.indexOf(item.globalPos) !== -1;
      });
    },
  },

  // ── Watchers (sync from props when parent updates server-side) ───────
  watch: {
    order: function(val) {
      if (val) this.localOrder = val.slice();
    },
    page: function(val) {
      this.localPage = val || 1;
    },
    pageSize: function(val) {
      this.localPageSize = val || 10;
      this.pageSizeEdit  = val || 10;
    },
    selectedPositions: function(val) {
      this.selectedArr = val ? val.slice() : [];
    },
    // Debounce search: wait 300ms after typing stops before filtering
    searchQuery: function(val) {
      var self = this;
      if (self.searchTimer) clearTimeout(self.searchTimer);
      self.searchTimer = setTimeout(function() {
        self.debouncedQuery = val;
        self.goPage(1);
      }, 300);
    },
    // Focus set-to input whenever the form becomes visible
    showSetTo: function(val) {
      if (val) {
        var self = this;
        this.$nextTick(function() {
          if (self.$refs.setToInput) self.$refs.setToInput.focus();
        });
      }
    },
  },

  // ── Methods ──────────────────────────────────────────────────────────
  methods: {

    // ── Drag & Drop ──────────────────────────────────────────────────

    onDragStart: function(evt, item) {
      this.isDragging = true;
      evt.dataTransfer.effectAllowed = 'move';
      evt.dataTransfer.setData('text/plain', String(item.globalPos));
      // If the dragged row is not already selected, track it separately.
      // We deliberately do NOT touch selectedArr here so:
      //  - the action panel does not pop up for an unselected drag
      //  - the checkbox next to the row does not flash selected mid-drag
      if (this.selectedArr.indexOf(item.globalPos) === -1) {
        this.dragItem = item;
      } else {
        this.dragItem = null;
      }
    },

    onDragOver: function(evt, item) {
      if (!this.isDragging) return;
      evt.preventDefault();
      // Only highlight if the hover target is not part of the selection
      if (this.selectedArr.indexOf(item.globalPos) === -1) {
        this.dragOverPos = item.globalPos;
      } else {
        this.dragOverPos = null;
      }
    },

    onDragLeave: function(evt) {
      // Only clear when actually leaving the row (not entering a child element)
      if (evt.relatedTarget && evt.currentTarget.contains(evt.relatedTarget)) return;
      this.dragOverPos = null;
    },

    onDrop: function(evt, targetItem) {
      evt.preventDefault();
      this.dragOverPos = null;
      this.isDragging  = false;

      // Determine what is being moved:
      //  isSingleDrag = true  → an unselected row was dragged; use dragItem
      //  isSingleDrag = false → one or more selected rows are being dragged
      var isSingleDrag = this.dragItem !== null;
      var positions;
      if (isSingleDrag) {
        // Drop onto self — nothing to do
        if (this.dragItem.globalPos === targetItem.globalPos) {
          this.dragItem = null;
          return;
        }
        positions     = [this.dragItem.globalPos];
        this.dragItem = null;
      } else {
        // Cannot drop onto a selected row
        if (this.selectedArr.indexOf(targetItem.globalPos) !== -1) return;
        positions = this.selectedArr.slice().sort(function(a, b) { return a - b; });
      }

      if (positions.length === 0) return;

      var toPosition = targetItem.globalPos;
      var newOrder   = this.reorderTo(this.localOrder, positions, toPosition);

      var itemsBeforeTarget = positions.filter(function(p) { return p < toPosition; }).length;
      var adjustedStart     = toPosition - itemsBeforeTarget;

      // For a selected-rows drag: move the selection to the new positions.
      // For a single unselected drag: leave the existing selection untouched.
      var newPositions = isSingleDrag
        ? this.selectedArr.slice()
        : positions.map(function(_, i) { return adjustedStart + i; });

      this.applyOrder(newOrder, newPositions);
      // Navigate to the page that now contains the dropped block
      var newPage = Math.floor(adjustedStart / this.localPageSize) + 1;
      this.goPage(Math.min(newPage, this.totalPages));
    },

    onDragEnd: function() {
      this.isDragging  = false;
      this.dragOverPos = null;
      this.dragItem    = null;
    },

    // ── Selection ────────────────────────────────────────────────────

    handleHeaderCheckbox: function() {
      // Flip: if all page-rows are selected → deselect all; otherwise → select all.
      // Driven purely from current state rather than trusting el-checkbox's emitted val.
      this.toggleAll(!this.pageAllSelected);
    },

    toggleSelect: function(globalPos) {
      var idx    = this.selectedArr.indexOf(globalPos);
      var newArr = this.selectedArr.slice();
      if (idx === -1) {
        newArr.push(globalPos);
      } else {
        newArr.splice(idx, 1);
      }
      this.selectedArr = newArr;
      this.$emit('update:selectedPositions', newArr.slice());
    },

    toggleAll: function(val) {
      var newArr = this.selectedArr.slice();
      this.pageItems.forEach(function(item) {
        var idx = newArr.indexOf(item.globalPos);
        if (val && idx === -1) {
          newArr.push(item.globalPos);
        } else if (!val && idx !== -1) {
          newArr.splice(idx, 1);
        }
      });
      this.selectedArr = newArr;
      this.$emit('update:selectedPositions', newArr.slice());
    },

    clearSelection: function() {
      this.selectedArr = [];
      this.$emit('update:selectedPositions', []);
    },

    // ── Core reorder helper ──────────────────────────────────────────
    //
    // Moves `positions` (sorted 0-based indices into `order`) so they appear
    // starting at `toPosition` (0-based index in the *original* order array).
    // Items in `positions` are kept in their current relative order.

    reorderTo: function(order, positions, toPosition) {
      var posSet = {};
      positions.forEach(function(p) { posSet[p] = true; });

      var sortedFrom = positions.slice().sort(function(a, b) { return a - b; });
      var movedItems = sortedFrom.map(function(p) { return order[p]; });
      var remaining  = order.filter(function(_, i) { return !posSet[i]; });

      // Adjust target: each selected item before `toPosition` shifts the
      // insertion point left by one in the `remaining` array.
      var itemsBeforeTarget = sortedFrom.filter(function(p) { return p < toPosition; }).length;
      var adjustedTarget    = Math.max(0, Math.min(toPosition - itemsBeforeTarget, remaining.length));

      var result = remaining.slice();
      // ES5-safe splice with spread equivalent
      Array.prototype.splice.apply(result, [adjustedTarget, 0].concat(movedItems));
      return result;
    },

    // Apply a new order, update selection, and notify parent + backend.
    applyOrder: function(newOrder, newPositions) {
      this.localOrder  = newOrder.slice();
      this.selectedArr = newPositions.slice();

      this.$emit('update:order',             newOrder.slice());
      this.$emit('update:selectedPositions', newPositions.slice());

      var self = this;
      this.$nextTick(function() {
        self.$emit('order-changed');
      });
    },

    // ── Action-panel operations ──────────────────────────────────────

    moveToTop: function() {
      var positions = this.selectedArr.slice().sort(function(a, b) { return a - b; });
      if (positions.length === 0) return;
      var newOrder = this.reorderTo(this.localOrder, positions, 0);
      var newPos   = positions.map(function(_, i) { return i; });
      this.applyOrder(newOrder, newPos);
      this.goPage(1);
    },

    moveUp: function() {
      var positions = this.selectedArr.slice().sort(function(a, b) { return a - b; });
      if (positions.length === 0) return;
      var first = positions[0];
      if (first === 0) return; // already at top
      var newOrder = this.reorderTo(this.localOrder, positions, first - 1);
      var newPos   = positions.map(function(_, i) { return first - 1 + i; });
      this.applyOrder(newOrder, newPos);
      // Jump to the page that now shows the first selected item
      this.goPage(Math.floor((first - 1) / this.localPageSize) + 1);
    },

    moveDown: function() {
      var positions = this.selectedArr.slice().sort(function(a, b) { return a - b; });
      if (positions.length === 0) return;
      var last = positions[positions.length - 1];
      if (last >= this.localOrder.length - 1) return; // already at bottom

      // Insert just after the item that currently sits below the group.
      var toPosition = last + 2;
      var newOrder   = this.reorderTo(this.localOrder, positions, toPosition);

      // All positions are <= last < toPosition, so all count as "before target".
      var adjustedStart = toPosition - positions.length;
      var newPos        = positions.map(function(_, i) { return adjustedStart + i; });
      this.applyOrder(newOrder, newPos);
      this.goPage(Math.min(Math.floor(adjustedStart / this.localPageSize) + 1, this.totalPages));
    },

    moveToBottom: function() {
      var positions = this.selectedArr.slice().sort(function(a, b) { return a - b; });
      if (positions.length === 0) return;
      var n        = this.localOrder.length;
      var newOrder = this.reorderTo(this.localOrder, positions, n);
      var start    = n - positions.length;
      var newPos   = positions.map(function(_, i) { return start + i; });
      this.applyOrder(newOrder, newPos);
      this.goPage(this.totalPages);
    },

    openSetTo: function() {
      this.setToValue = 1;
      this.showSetTo  = true;
      // Watch handler will focus the input after the next tick
    },

    applySetTo: function() {
      var positions = this.selectedArr.slice().sort(function(a, b) { return a - b; });
      if (positions.length === 0) { this.showSetTo = false; return; }

      var targetPos  = Math.max(1, Math.min(Math.round(this.setToValue) || 1, this.localOrder.length));
      var toPosition = targetPos - 1; // convert to 0-based

      var newOrder              = this.reorderTo(this.localOrder, positions, toPosition);
      var itemsBeforeTarget     = positions.filter(function(p) { return p < toPosition; }).length;
      var adjustedStart         = toPosition - itemsBeforeTarget;
      var newPos                = positions.map(function(_, i) { return adjustedStart + i; });

      this.applyOrder(newOrder, newPos);
      this.goPage(Math.min(Math.floor(adjustedStart / this.localPageSize) + 1, this.totalPages));
      this.showSetTo = false;
    },

    // ── Pagination ───────────────────────────────────────────────────

    goPage: function(p) {
      var clamped   = Math.max(1, Math.min(Math.round(p), this.totalPages));
      this.localPage = clamped;
      this.$emit('update:page', clamped);
    },

    startEditPageSize: function() {
      this.pageSizeEdit    = this.localPageSize;
      this.editingPageSize = true;
      var self = this;
      this.$nextTick(function() {
        if (self.$refs.pageSizeInput) self.$refs.pageSizeInput.focus();
      });
    },

    cancelEditPageSize: function() {
      this.editingPageSize = false;
    },

    confirmPageSize: function() {
      var val           = Math.max(1, Math.round(this.pageSizeEdit) || 10);
      this.localPageSize = val;
      this.editingPageSize = false;
      this.$emit('update:pageSize', val);
      // Re-clamp using totalPages, which is derived from filteredOrder (not localOrder)
      this.goPage(this.localPage);
    },
  },
});
