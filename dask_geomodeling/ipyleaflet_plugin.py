try:
    from notebook.utils import url_path_join
    from notebook.base.handlers import IPythonHandler
    from ipyleaflet import WMSLayer
except ImportError:
    url_path_join = None
    IPythonHandler = object
    WMSLayer = None
    raise

from dask_geomodeling.core import Block
from datetime import datetime
from matplotlib import cm
from matplotlib.colors import Normalize
from io import BytesIO
import numpy as np
from PIL import Image

# include this plugin with:
# jupyter notebook --NotebookApp.nbserver_extensions="{'dask_geomodeling.ipyleaflet_plugin':True}"
class HelloWorldHandler(IPythonHandler):
    def get(self):
        self.finish("Hello world!")


class GeomodelingWMSHandler(IPythonHandler):
    def get(self):
        # the idea is to get the geomodeling config + WMS / TMS params from
        # the request params to set up a WMS / TMS service
        block = Block.from_json(self.get_query_argument("layers"))
        style = self.get_query_argument("styles")
        format = self.get_query_argument("format")
        transparent = self.get_query_argument("transparent")
        srs = self.get_query_argument("srs")
        height = int(self.get_query_argument("height"))
        max_cell_size = 5  # TODO
        time = datetime.utcnow()  # TODO datetime.fromisoformat(self.get_query_argument("time"))
        width = int(self.get_query_argument("width"))
        bbox = [float(x) for x in self.get_query_argument("bbox").split(",")]

        # overload protection
        cell_size_x = (bbox[2] - bbox[0]) / width
        cell_size_y = (bbox[3] - bbox[1]) / height
        if cell_size_x > max_cell_size or cell_size_y > max_cell_size:
            self.finish("Too large area requested")
            return

        # get cmap
        cmap, vmin, vmax, alpha = style.split(":")

        data = block.get_data(
            mode="vals",
            bbox=bbox,
            height=height,
            width=width,
            projection=srs,
            start=time,
        )
        masked = np.ma.masked_equal(data["values"][0], data["no_data_value"])
        stream = BytesIO()

        normalized = Normalize(vmin=float(vmin), vmax=float(vmax))(masked)
        img = cm.get_cmap(cmap)(normalized)
        img[:, :, 3] = float(alpha)
        img[normalized.mask, 3] = 0.
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(stream, format="png")
        raw = stream.getvalue()

        self.set_header("Content-Length", len(raw))
        self.set_header("Content-Type", "image/png")
        self.set_header("Pragma", "no-cache")
        self.set_header("Cache-Control",
                        "no-store, "
                        "no-cache=Set-Cookie, "
                        "proxy-revalidate, "
                        "max-age=0, "
                        "post-check=0, pre-check=0")
        self.set_header("Expires", "Wed, 2 Dec 1837 21:00:12 GMT")
        self.write(raw)
        self.finish()


class GeomodelingLayer(WMSLayer):
    base_url = None  # set by load_jupyter_server_extension

    def __init__(self, hostname="localhost", **kwargs):
        from notebook import notebookapp

        self.format = "image/png"

        for server in notebookapp.list_running_servers():
            if server["hostname"] == hostname:
                kwargs["url"] = server["url"] + "wms"
                break

        super().__init__(**kwargs)

    def set_block(self, block):
        self.layers = block.to_json()


def load_jupyter_server_extension(nb_server_app):
    """
    Called when the extension is loaded.

    Args:
        nb_server_app (NotebookWebApplication): handle to the Notebook webserver instance.
    """
    web_app = nb_server_app.web_app
    host_pattern = '.*$'
    route_pattern = url_path_join(web_app.settings['base_url'], '/hello')
    web_app.add_handlers(host_pattern, [(route_pattern, HelloWorldHandler)])
    route_pattern = url_path_join(web_app.settings['base_url'], '/wms')
    web_app.add_handlers(host_pattern, [(route_pattern, GeomodelingWMSHandler)])
