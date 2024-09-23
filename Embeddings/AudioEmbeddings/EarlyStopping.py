# Keeps track of the validation loss and triggers early stopping when applicable
# Patience indicates how many times the loss can get worse before stopping
class EarlyStopping(object):
    def __init__(self, patience=50):
        self.patience = patience  # how many times to let a bad loss through in a row
        self.lowest_loss = None  # previous loss
        self.num_trigger = 0  # how many times a bad loss has happened in a row

    def stop(self, loss):
        # Not seen anything yet, initialise
        if self.lowest_loss == None:
            self.lowest_loss = loss
        # New best loss
        elif loss < self.lowest_loss:
            self.lowest_loss = loss
            self.num_trigger = 0
        # Otherwise add to the trigger
        else:
            self.num_trigger += 1
            # If triggered too many times, stop
            if self.num_trigger >= self.patience:
                return True
        # Survived, keep going!
        return False