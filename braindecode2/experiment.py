from collections import OrderedDict
import logging
import pandas as pd

from braindecode2.torchext.util import to_net_in_output, set_random_seeds

log = logging.getLogger(__name__)

class Experiment(object):
    def __init__(self, model, train_set, valid_set, test_set,
                 iterator, loss_function, optimizer, updates_modifier,
                 monitors, stop_criterion, remember_best_chan,
                 run_after_early_stop,
                 batch_modifier=None, cuda=True):
        self.model = model
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', valid_set), ('test', test_set)))
        self.iterator = iterator
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.updates_modifier = updates_modifier
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        # self.monitor_manager = MonitorManager(monitors)
        # self.remember_extension = RememberBest(remember_best_chan)
        self.run_after_early_stop = run_after_early_stop
        self.batch_modifier = batch_modifier
        self.cuda = cuda
        self.epochs_df = pd.DataFrame()

    def run(self):
        self.setup_training()
        log.info("Run until first stop...")
        self.run_until_early_stop()
        # always setup for second stop, in order to get best model
        # even if not running after early stop...
        log.info("Setup for second stop...")
        self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Run until second stop...")
            self.run_until_second_stop()
            self.readd_old_monitor_chans()

    def setup_training(self):
        # reset remember best extension in case you rerun some experiment
        # self.remember_extension = RememberBest(
        #    self.remember_extension.chan_name)
        # log.info("Done.")
        set_random_seeds(seed=2382938, cuda=self.cuda)
        if self.cuda:
            self.model.cuda()

    def run_until_early_stop(self):
        #self.create_monitors(datasets)
        self.run_until_stop(remember_best=True)

    def run_until_stop(self, datasets, remember_best):
        self.monitor_epoch(datasets)
        self.print_epoch()
        # if remember_best:
        #    self.remember_extension.remember_epoch(self.monitor_chans,
        #        self.all_params)

        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(datasets, remember_best)

    def run_one_epoch(self, datasets, remember_best):
        batch_generator = self.iterator.get_batches(datasets['train'],
                                                    shuffle=True)
        # TODO, add timing again?
        for inputs, targets in batch_generator:
            if self.batch_modifier is not None:
                inputs, targets = self.batch_modifier.process(inputs,
                                                              targets)
            # could happen that batch modifier has removed all inputs...
            if len(inputs) > 0:
                self.train_batch(inputs, targets)

        self.monitor_epoch(datasets)
        self.print_epoch()
        # if remember_best:
        #    self.remember_extension.remember_epoch(self.monitor_chans,
        #        self.all_params)

    def monitor_epoch(self, datasets):
        row_dict = OrderedDict()
        for m in self.monitors:
            result_dict =  m.monitor_epoch()
            if result_dict is not None:
                row_dict.update(result_dict)
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            all_losses = []
            batch_sizes = []
            targets = []
            for batch in self.iterator.get_batches(dataset, shuffle=False):
                preds, loss = self.eval_on_batch(batch[0], batch[1])
                all_preds.append(preds)
                all_losses.append(loss)
                batch_sizes.append(len(batch[0]))
                targets.append(batch[1])

            for m in self.monitors:
                result_dict = m.monitor_set(setname, all_preds, all_losses,
                    batch_sizes, targets, dataset)
                if result_dict is not None:
                    row_dict.update(result_dict)
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
        assert set(self.epochs_df.columns) == set(row_dict.keys())
        self.epochs_df = self.epochs_df[list(row_dict.keys())]

    def print_epoch(self):
        # -1 due to doing one monitor at start of training
        i_epoch = len(self.epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = self.epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            log.info("{:25s} {:.5f}".format(key, val))
        log.info("")

    def eval_on_batch(self, inputs, targets):
        self.model.eval()
        input_vars = to_net_in_output(inputs)
        target_vars = to_net_in_output(targets)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        outputs = outputs.cpu().data.numpy()
        loss = loss.cpu().data.numpy()
        return outputs, loss

    def train_batch(self, inputs, targets):
        self.model.train()
        input_vars = to_net_in_output(inputs)
        target_vars = to_net_in_output(targets)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        loss.backward()
        self.optimizer.step()


    def setup_after_stop_training(self):
        # also remember old monitor chans, will be put back into
        # monitor chans after experiment finished
        self.old_monitor_chans = deepcopy(self.monitor_chans)
        self.remember_extension.reset_to_best_model(self.monitor_chans,
                                                    self.all_params)
        loss_to_reach = self.monitor_chans['train_loss'][-1]
        self.stop_criterion = Or(stop_criteria=[
            MaxEpochs(num_epochs=self.remember_extension.best_epoch * 2),
            ChanBelow(chan_name='valid_loss', target_value=loss_to_reach)])
        log.info("Train loss to reach {:.5f}".format(loss_to_reach))

    def run_until_second_stop(self):
        datasets = self.dataset_provider.get_train_merged_valid_test(
            self.dataset)
        self.run_until_stop(datasets, remember_best=False)

    def create_monitors(self, datasets):
        self.monitor_chans = OrderedDict()
        self.last_epoch_time = None
        for monitor in self.monitors:
            monitor.setup(self.monitor_chans, datasets)


    def readd_old_monitor_chans(self):
        for key in self.old_monitor_chans:
            new_key = 'before_reset_' + key
            self.monitor_chans[new_key] = self.old_monitor_chans[key]