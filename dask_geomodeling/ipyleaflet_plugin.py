try:
    from notebook.utils import url_path_join
    from notebook.base.handlers import IPythonHandler
except ImportError:
    url_path_join = None
    IPythonHandler = object


# include this plugin with:
# jupyter notebook --NotebookApp.nbserver_extensions="{'dask_geomodeling.ipyleaflet_plugin':True}"

class HelloWorldHandler(IPythonHandler):
    def get(self):
        # the idea is to get the geomodeling config + WMS / TMS params from
        # the request params to set up a WMS / TMS service
        self.finish(str(len(self.get_argument("bla"))))


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
