const fileInput = document.getElementById('file');
const dropZone = document.getElementById('drop-zone');
let previewUrl;

function showPreview() {
  const file = fileInput.files[0];
  fileInput.setCustomValidity('');
  if (!file) return;

  if (file.type && !['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    fileInput.setCustomValidity('Choose a PNG, JPEG, or WebP image.');
  } else if (file.size > Number(fileInput.dataset.maxBytes)) {
    fileInput.setCustomValidity('This image is too large. Resize it and try again.');
  }
  if (!fileInput.reportValidity()) return;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  const preview = document.getElementById('board-preview');
  preview.src = previewUrl;
  preview.alt = 'Preview of your selected chessboard image';
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('preview-label').textContent = 'Your board';
  document.getElementById('preview-note').textContent = 'Check the turn and board orientation before analysing.';
}

if (fileInput) {
  fileInput.addEventListener('change', showPreview);
  dropZone.addEventListener('dragover', event => {
    event.preventDefault();
    dropZone.classList.add('dragging');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', event => {
    event.preventDefault();
    dropZone.classList.remove('dragging');
    if (event.dataTransfer.files.length) {
      fileInput.files = event.dataTransfer.files;
      showPreview();
    }
  });
}

document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    button.disabled = true;
    button.textContent = 'Analysing...';
  });
});

window.addEventListener('pageshow', event => {
  if (event.persisted) window.location.reload();
});

const copyButton = document.getElementById('copy-fen');
if (copyButton) {
  copyButton.addEventListener('click', async () => {
    const fen = document.getElementById('fen');
    try {
      await navigator.clipboard.writeText(fen.value);
      copyButton.textContent = 'Copied';
      setTimeout(() => { copyButton.textContent = 'Copy FEN'; }, 1500);
    } catch {
      fen.select();
      copyButton.textContent = 'Press Ctrl+C to copy';
    }
  });
}
