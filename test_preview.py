"""Minimal test: webcam -> GTK4 preview window."""
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gst", "1.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, Adw, Gst, GLib, Gdk, Gio
import threading

Gst.init(None)

class TestApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.test.preview")
        self._pipeline = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._width = 1280
        self._height = 720

    def do_activate(self):
        win = Adw.ApplicationWindow(application=self, title="Preview Test")
        win.set_default_size(800, 500)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)

        self._picture = Gtk.Picture()
        self._picture.set_hexpand(True)
        self._picture.set_vexpand(True)
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        box.append(self._picture)

        self._label = Gtk.Label(label="Starting...")
        box.append(self._label)

        win.set_content(box)
        win.present()

        # Build pipeline
        self._pipeline = Gst.parse_launch(
            'v4l2src device=/dev/video0 ! '
            'image/jpeg,width=1280,height=720,framerate=30/1 ! '
            'jpegdec ! videoconvert ! '
            'video/x-raw,format=BGRA,width=1280,height=720 ! '
            'appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false'
        )
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._label.set_text("Pipeline started")

        # Poll for frames on main thread
        GLib.timeout_add(33, self._update_preview)

    def _on_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            with self._lock:
                self._latest_frame = bytes(info.data)
            buf.unmap(info)
        return Gst.FlowReturn.OK

    def _update_preview(self):
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None

        if frame and len(frame) == self._width * self._height * 4:
            gbytes = GLib.Bytes.new(frame)
            texture = Gdk.MemoryTexture.new(
                self._width, self._height,
                Gdk.MemoryFormat.B8G8R8A8,
                gbytes, self._width * 4
            )
            self._picture.set_paintable(texture)
            self._label.set_text(f"Rendering ({len(frame)} bytes)")
        return True

app = TestApp()
app.run(None)
