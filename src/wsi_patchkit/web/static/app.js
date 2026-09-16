(() => {
  "use strict";

  const menuButton = document.querySelector("#slide-menu-button");
  const menu = document.querySelector("#slide-menu");
  const currentSlideLabel = document.querySelector("#current-slide");
  const slideFilter = document.querySelector("#slide-filter");
  const slideCount = document.querySelector("#slide-count");
  const slideList = document.querySelector("#slide-list");
  const slideEmpty = document.querySelector("#slide-empty");
  const slider = document.querySelector("#zoom-slider");
  const zoomLabel = document.querySelector("#zoom-label");
  const coordinateLabel = document.querySelector("#coordinate-label");
  const emptyState = document.querySelector("#empty-state");
  const statusMessage = document.querySelector("#status-message");
  const scale = document.querySelector("#scale");
  const scaleLabel = document.querySelector("#scale-label");
  const scaleBar = document.querySelector("#scale-bar");

  let slides = new Map();
  let currentSlide = null;
  let viewer = null;
  let openSequence = 0;

  function showStatus(message, loading = false) {
    statusMessage.textContent = message;
    emptyState.hidden = false;
    const spinner = emptyState.querySelector(".spinner");
    spinner.hidden = !loading;
  }

  function hideStatus() {
    emptyState.hidden = true;
  }

  function currentItem() {
    return viewer && viewer.world.getItemCount() ? viewer.world.getItemAt(0) : null;
  }

  function niceScaleLength(targetMicrometres) {
    const exponent = 10 ** Math.floor(Math.log10(targetMicrometres));
    const normalized = targetMicrometres / exponent;
    const step = normalized < 2 ? 1 : normalized < 5 ? 2 : 5;
    return step * exponent;
  }

  function updateViewportStatus() {
    const item = currentItem();
    if (!item) return;

    const imageZoom = item.viewportToImageZoom(viewer.viewport.getZoom(true));
    slider.value = String(Math.log2(imageZoom));

    if (!currentSlide?.mpp) {
      zoomLabel.textContent = `缩放 ${(imageZoom * 100).toFixed(0)}%`;
      scale.hidden = true;
      return;
    }

    const baseMpp = (currentSlide.mpp[0] + currentSlide.mpp[1]) / 2;
    const screenMpp = baseMpp / imageZoom;
    zoomLabel.textContent =
      `缩放 ${(imageZoom * 100).toFixed(0)}% · ${screenMpp.toFixed(3)} µm/px`;

    const physicalLength = niceScaleLength(screenMpp * 120);
    const barWidth = Math.max(45, Math.min(180, physicalLength / screenMpp));
    scaleBar.style.width = `${barWidth}px`;
    scaleLabel.textContent =
      physicalLength >= 1000
        ? `${(physicalLength / 1000).toLocaleString()} mm`
        : `${physicalLength.toLocaleString()} µm`;
    scale.hidden = false;
  }

  function setImageZoom(imageZoom) {
    const item = currentItem();
    if (!item) return;
    const viewportZoom = item.imageToViewportZoom(imageZoom);
    viewer.viewport.zoomTo(viewportZoom);
    viewer.viewport.applyConstraints();
  }

  async function openSlide(slideId) {
    const record = slides.get(slideId);
    if (!record) return;
    const sequence = ++openSequence;
    currentSlideLabel.textContent = slideId;
    for (const item of slideList.querySelectorAll("button")) {
      const selected = item.dataset.slideId === slideId;
      item.classList.toggle("selected", selected);
      item.setAttribute("aria-current", selected ? "true" : "false");
    }
    closeSlideMenu();
    showStatus(`正在打开 ${slideId}…`, true);
    coordinateLabel.textContent = "x — · y —";
    scale.hidden = true;
    try {
      let metadata = record;
      if (metadata.width === undefined) {
        const response = await fetch(`/api/slides/${encodeURIComponent(slideId)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        metadata = await response.json();
        slides.set(slideId, metadata);
      }
      if (sequence !== openSequence) return;
      currentSlide = metadata;
      viewer.open(`/iiif/3/${encodeURIComponent(slideId)}/info.json`);
    } catch (error) {
      if (sequence === openSequence) {
        showStatus(`切片加载失败：${error.message}`);
      }
    }
  }

  function closeSlideMenu() {
    menu.hidden = true;
    menuButton.setAttribute("aria-expanded", "false");
  }

  function setSlideMenuOpen(open) {
    menu.hidden = !open;
    menuButton.setAttribute("aria-expanded", String(open));
    if (open) {
      slideFilter.focus();
      const selected = slideList.querySelector("button.selected");
      selected?.scrollIntoView({ block: "nearest" });
    }
  }

  function filterSlides() {
    const query = slideFilter.value.trim().toLocaleLowerCase();
    let visible = 0;
    for (const item of slideList.children) {
      const matches = item.textContent.toLocaleLowerCase().includes(query);
      item.hidden = !matches;
      if (matches) visible += 1;
    }
    slideCount.textContent = query
      ? `${visible} / ${slides.size} 张切片`
      : `${slides.size} 张切片`;
    slideEmpty.hidden = visible !== 0;
  }

  function populateSlideMenu(records) {
    const fragment = document.createDocumentFragment();
    for (const record of records) {
      const row = document.createElement("li");
      row.setAttribute("role", "none");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.slideId = record.id;
      button.setAttribute("role", "menuitem");
      button.textContent = record.id;
      button.title = record.id;
      button.addEventListener("click", () => openSlide(record.id));
      row.append(button);
      fragment.append(row);
    }
    slideList.replaceChildren(fragment);
    filterSlides();
  }

  async function initialize() {
    if (!window.OpenSeadragon) {
      showStatus("OpenSeadragon 加载失败，请检查网络连接");
      return;
    }

    viewer = OpenSeadragon({
      id: "viewer",
      prefixUrl:
        "https://cdn.jsdelivr.net/npm/openseadragon@5.0.1/build/openseadragon/images/",
      showNavigator: true,
      navigatorPosition: "TOP_RIGHT",
      navigatorSizeRatio: 0.16,
      showNavigationControl: false,
      animationTime: 0.45,
      springStiffness: 8,
      zoomPerScroll: 1.25,
      minZoomImageRatio: 0.8,
      maxZoomPixelRatio: 2,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
      gestureSettingsMouse: {
        scrollToZoom: true,
        dragToPan: true,
        clickToZoom: false,
        dblClickToZoom: true,
      },
      gestureSettingsTouch: {
        pinchToZoom: true,
        dragToPan: true,
        flickEnabled: true,
        dblClickToZoom: true,
      },
    });

    viewer.addHandler("open", () => {
      hideStatus();
      updateViewportStatus();
    });
    viewer.addHandler("open-failed", (event) => {
      showStatus(`切片加载失败：${event.message || "未知错误"}`);
    });
    viewer.addHandler("viewport-change", updateViewportStatus);
    viewer.addHandler("canvas-hover", (event) => {
      const item = currentItem();
      if (!item || !event.position) return;
      const viewportPoint = viewer.viewport.pointFromPixel(event.position);
      const imagePoint = item.viewportToImageCoordinates(viewportPoint);
      const inBounds =
        imagePoint.x >= 0 &&
        imagePoint.y >= 0 &&
        imagePoint.x < currentSlide.width &&
        imagePoint.y < currentSlide.height;
      coordinateLabel.textContent = inBounds
        ? `x ${Math.floor(imagePoint.x).toLocaleString()} · y ${Math.floor(imagePoint.y).toLocaleString()}`
        : "x — · y —";
    });

    document.querySelector("#zoom-in").addEventListener("click", () => {
      viewer.viewport.zoomBy(1.5);
      viewer.viewport.applyConstraints();
    });
    document.querySelector("#zoom-out").addEventListener("click", () => {
      viewer.viewport.zoomBy(1 / 1.5);
      viewer.viewport.applyConstraints();
    });
    document.querySelector("#home-view").addEventListener("click", () => {
      viewer.viewport.goHome();
    });
    slider.addEventListener("input", () => setImageZoom(2 ** Number(slider.value)));
    menuButton.addEventListener("click", () => setSlideMenuOpen(menu.hidden));
    slideFilter.addEventListener("input", filterSlides);
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".slide-picker")) closeSlideMenu();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !menu.hidden) {
        closeSlideMenu();
        menuButton.focus();
      }
    });

    try {
      const response = await fetch("/api/slides");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const records = await response.json();
      if (!records.length) throw new Error("服务端没有注册切片");
      slides = new Map(records.map((record) => [record.id, record]));
      populateSlideMenu(records);
      openSlide(records[0].id);
    } catch (error) {
      showStatus(`无法读取切片列表：${error.message}`);
    }
  }

  window.addEventListener("DOMContentLoaded", initialize);
})();
