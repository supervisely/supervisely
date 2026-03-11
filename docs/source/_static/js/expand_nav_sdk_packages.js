(function () {
  function isSdkPackagesPage() {
    return /(^|\/)sdk_packages\.html$/.test(window.location.pathname);
  }

  function expandAllNavigationSections() {
    var sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return;

    var toggles = sidebar.querySelectorAll("input.md-nav__toggle");
    toggles.forEach(function (el) {
      el.checked = true;
    });
  }

  if (!isSdkPackagesPage()) return;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", expandAllNavigationSections);
  } else {
    expandAllNavigationSections();
  }
})();

