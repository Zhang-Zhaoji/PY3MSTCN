class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name = None):
        self.reset()
        if name is not None:
            self.name = name
        else:
            self.name = "DefaultAverageMeter"

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}:{self.avg:.6f} = {self.sum:.6f}/{self.count:.6f}'
