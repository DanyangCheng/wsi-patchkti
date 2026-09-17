(() => {
  "use strict";

  const cropSizeStorageKey = "wsi-patchkit.crop-size";
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
  const cropTool = document.querySelector("#crop-tool");
  const cropPanel = document.querySelector("#crop-panel");
  const cropClose = document.querySelector("#crop-close");
  const cropX = document.querySelector("#crop-x");
  const cropY = document.querySelector("#crop-y");
  const cropWidth = document.querySelector("#crop-width");
  const cropHeight = document.querySelector("#crop-height");
  const cropFormat = document.querySelector("#crop-format");
  const cropFilename = document.querySelector("#crop-filename");
  const cropSave = document.querySelector("#crop-save");
  const cropStatus = document.querySelector("#crop-status");
  const cropOverlay = document.querySelector("#crop-overlay");
  const cropOverlaySize = document.querySelector("#crop-overlay-size");

  let slides = new Map();
  let currentSlide = null;
  let viewer = null;
  let openSequence = 0;
  let cropActive = false;
  let cropOverlayAdded = false;

  function loadCropSize() {
    try {
      const value = JSON.parse(localStorage.getItem(cropSizeStorageKey));
      if (
        Number.isInteger(value?.width) &&
        value.width > 0 &&
        Number.isInteger(value?.height) &&
        value.height > 0
      ) {
        return value;
      }
    } catch (_) {
      // Storage may be unavailable or contain data from an older version.
    }
    return { width: 1024, height: 1024 };
  }

  function saveCropSize(size) {
    try {
      localStorage.setItem(cropSizeStorageKey, JSON.stringify(size));
    } catch (_) {
      // Keep the in-memory preference when persistent storage is unavailable.
    }
  }

  let preferredCropSize = loadCropSize();
  let cropRegion = { x: 0, y: 0, ...preferredCropSize };

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

  function setCropStatus(message, type = "") {
    cropStatus.textContent = message;
    cropStatus.className = `crop-status ${type}`.trim();
  }

  function updateCropOverlay() {
    const item = currentItem();
    if (!item || !cropActive) return;
    const imageRect = new OpenSeadragon.Rect(
      cropRegion.x,
      cropRegion.y,
      cropRegion.width,
      cropRegion.height,
    );
    const viewportRect = item.imageToViewportRectangle(imageRect);
    cropOverlay.hidden = false;
    cropOverlaySize.textContent =
      `${cropRegion.width.toLocaleString()} × ${cropRegion.height.toLocaleString()} px`;
    if (cropOverlayAdded) {
      viewer.updateOverlay(cropOverlay, viewportRect);
    } else {
      viewer.addOverlay({ element: cropOverlay, location: viewportRect });
      cropOverlayAdded = true;
    }
  }

  function setCropRegion(region, updateInputs = true) {
    if (!currentSlide) return;
    const width = Math.max(1, Math.min(Math.round(region.width), currentSlide.width));
    const height = Math.max(1, Math.min(Math.round(region.height), currentSlide.height));
    cropRegion = {
      x: Math.max(0, Math.min(Math.round(region.x), currentSlide.width - width)),
      y: Math.max(0, Math.min(Math.round(region.y), currentSlide.height - height)),
      width,
      height,
    };
    if (updateInputs) {
      cropX.value = String(cropRegion.x);
      cropY.value = String(cropRegion.y);
      cropWidth.value = String(cropRegion.width);
      cropHeight.value = String(cropRegion.height);
      cropX.max = String(currentSlide.width - cropRegion.width);
      cropY.max = String(currentSlide.height - cropRegion.height);
      cropWidth.max = String(currentSlide.width);
      cropHeight.max = String(currentSlide.height);
    }
    updateCropOverlay();
  }

  function centerCropAt(imagePoint) {
    setCropRegion({
      ...cropRegion,
      x: imagePoint.x - cropRegion.width / 2,
      y: imagePoint.y - cropRegion.height / 2,
    });
  }

  function initializeCropRegion() {
    const item = currentItem();
    if (!item || !currentSlide) return;
    const center = item.viewportToImageCoordinates(viewer.viewport.getCenter(true));
    const width = Math.min(preferredCropSize.width, currentSlide.width);
    const height = Math.min(preferredCropSize.height, currentSlide.height);
    setCropRegion({
      x: center.x - width / 2,
      y: center.y - height / 2,
      width,
      height,
    });
  }

  function setCropActive(active) {
    cropActive = Boolean(active && currentItem() && currentSlide);
    cropTool.setAttribute("aria-pressed", String(cropActive));
    cropPanel.hidden = !cropActive;
    if (cropActive) {
      initializeCropRegion();
      setCropStatus("");
    } else {
      if (cropOverlayAdded) viewer.removeOverlay(cropOverlay);
      cropOverlayAdded = false;
      cropOverlay.hidden = true;
    }
  }

  function syncCropInputs() {
    const values = [cropX, cropY, cropWidth, cropHeight].map((input) =>
      Number(input.value),
    );
    if (!values.every(Number.isFinite)) return;
    setCropRegion({
      x: values[0],
      y: values[1],
      width: values[2],
      height: values[3],
    });
    preferredCropSize = {
      width: cropRegion.width,
      height: cropRegion.height,
    };
    saveCropSize(preferredCropSize);
    setCropStatus("");
  }

  async function saveCrop() {
    if (!currentSlide || !cropActive) return;
    cropSave.disabled = true;
    setCropStatus("正在生成 level-0 裁剪并保存…");
    try {
      const response = await fetch(
        `/api/slides/${encodeURIComponent(currentSlide.id)}/crops`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...cropRegion,
            format: cropFormat.value,
            filename: cropFilename.value.trim() || null,
          }),
        },
      );
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      cropFilename.value = "";
      setCropStatus(`已保存到服务器：${result.filename}`, "success");
    } catch (error) {
      setCropStatus(`保存失败：${error.message}`, "error");
    } finally {
      cropSave.disabled = false;
    }
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
      if (cropActive) initializeCropRegion();
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
    viewer.addHandler("canvas-click", (event) => {
      if (!cropActive || !event.quick || !event.position) return;
      const item = currentItem();
      if (!item) return;
      const viewportPoint = viewer.viewport.pointFromPixel(event.position);
      const imagePoint = item.viewportToImageCoordinates(viewportPoint);
      centerCropAt(imagePoint);
      event.preventDefaultAction = true;
    });

    new OpenSeadragon.MouseTracker({
      element: cropOverlay,
      clickHandler: (event) => {
        event.preventDefaultAction = true;
      },
      dragHandler: (event) => {
        const item = currentItem();
        if (!item || !cropActive) return;
        const viewportDelta = viewer.viewport.deltaPointsFromPixels(event.delta);
        const imageOrigin = item.viewportToImageCoordinates(
          new OpenSeadragon.Point(0, 0),
        );
        const imageDeltaPoint = item.viewportToImageCoordinates(viewportDelta);
        setCropRegion({
          ...cropRegion,
          x: cropRegion.x + imageDeltaPoint.x - imageOrigin.x,
          y: cropRegion.y + imageDeltaPoint.y - imageOrigin.y,
        });
        event.preventDefaultAction = true;
      },
    }).setTracking(true);

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
    cropTool.addEventListener("click", () => setCropActive(!cropActive));
    cropClose.addEventListener("click", () => setCropActive(false));
    for (const input of [cropX, cropY, cropWidth, cropHeight]) {
      input.addEventListener("change", syncCropInputs);
    }
    cropSave.addEventListener("click", saveCrop);
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
      } else if (event.key === "Escape" && cropActive) {
        setCropActive(false);
        cropTool.focus();
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
