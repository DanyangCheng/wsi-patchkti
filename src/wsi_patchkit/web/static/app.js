(() => {
  "use strict";

  const select = document.querySelector("#slide-select");
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

  function openSlide(slideId) {
    currentSlide = slides.get(slideId);
    if (!currentSlide) return;
    showStatus(`正在打开 ${slideId}…`, true);
    coordinateLabel.textContent = "x — · y —";
    scale.hidden = true;
    viewer.open(`/iiif/3/${encodeURIComponent(slideId)}/info.json`);
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
    select.addEventListener("change", () => openSlide(select.value));

    try {
      const response = await fetch("/api/slides");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const records = await response.json();
      if (!records.length) throw new Error("服务端没有注册切片");
      slides = new Map(records.map((record) => [record.id, record]));
      for (const record of records) {
        const option = document.createElement("option");
        option.value = record.id;
        option.textContent = record.id;
        select.append(option);
      }
      openSlide(records[0].id);
    } catch (error) {
      showStatus(`无法读取切片列表：${error.message}`);
    }
  }

  window.addEventListener("DOMContentLoaded", initialize);
})();

