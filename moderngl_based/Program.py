class Program:
    """Base class for shader programs."""

    def __init__(self, ctx):
        self.ctx = ctx


    @property
    def memUsedCpu(self):
        """Estimated amount of CPU-side memory used."""
        return None # amount not known

    @property
    def memUsedGpu(self):
        """Estimated amount of GPU-side memory used."""
        return None # amount not known


    def run(self, *args, **kwargs):
        """Execute the program."""
        raise NotImplementedError