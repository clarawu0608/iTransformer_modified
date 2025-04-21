from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self):

        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, ~mask)
        masked_true = torch.masked_select(y_true, ~mask)

        return self.mse_loss(masked_pred, masked_true)

def linear_interpolate_impute(input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    Imputes missing values in the input tensor using linear interpolation.
    Missing values are identified by a mask tensor, where 0 indicates missing.

    Args:
        input_tensor (torch.Tensor): Tensor of shape (time_points, features).
        mask_tensor (torch.Tensor): Binary tensor of shape (time_points, features),
                                    where 1 means observed and 0 means missing.
    
    Returns:
        torch.Tensor: Imputed tensor with missing values filled via linear interpolation.
    """
    input_np = input_tensor.numpy()
    mask_np = mask_tensor.numpy()
    
    time_points, features = input_np.shape
    imputed_np = input_np.copy()
    
    for feature in range(features):
        x = np.arange(time_points)
        y = input_np[:, feature]
        mask = mask_np[:, feature].astype(bool)
        
        if np.all(mask):  # No missing values
            continue
        
        if len(x[mask]) == 0 or len(y[mask]) == 0:
            imputed_values = x
        else:
            imputed_values = np.interp(x, x[mask], y[mask])
        imputed_np[:, feature] = imputed_values
    
    return torch.tensor(imputed_np, dtype=input_tensor.dtype)

def inverse_multi_tokens_array(tensor_array, n, m, T):
    """
    Reconstructs original matrices from (p, n, x*N) tokenized form.

    Args:
        tensor_array: numpy array of shape (p, n, x*N), dtype bool/int/float
        n: segment length
        m: overlap length
        T: desired output sequence length

    Returns:
        numpy array of shape (p, T, N)
    """
    p, _, total_dim = tensor_array.shape
    step = n - m
    x = (T + m) // n  # number of segments
    N = total_dim // x          # correct feature dimension

    is_bool_input = tensor_array.dtype == np.bool_
    is_int_input = np.issubdtype(tensor_array.dtype, np.integer)

    outputs = []

    for i in range(p):
        matrix = tensor_array[i]  # shape (n, x*N)
        segments = np.split(matrix, x, axis=1)         # list of (n, N)
        segments = np.stack([seg for seg in segments], axis=0)  # (x, n, N)


        if is_bool_input:
            output = np.zeros((T, N), dtype=bool)
            for j, seg in enumerate(segments):
                start = j * step
                end = start + n
                output[start:end] = seg  # overwrite
        else:
            output = np.zeros((T, N), dtype=np.float32)
            count = np.zeros((T, N), dtype=np.float32)
            for j, seg in enumerate(segments):
                start = j * step
                end = start + n
                output[start:end] += seg.astype(np.float32)
                count[start:end] += 1
            count[count == 0] = 1
            output = output / count
            if is_int_input:
                output = np.rint(output).astype(tensor_array.dtype)

        outputs.append(output[:T])  # Trim to exact T

    return np.stack(outputs)

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        self.args.pred_len = self.args.seq_len
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (not be used in iTransformer)
                dec_inp = batch_x

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0 # the last one or all of them
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        vali_losses = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = batch_x

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            train_losses.append(train_loss)
            vali_losses.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        # Plot training and validation loss
        folder_path = f'./results/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(len(vali_losses)), vali_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./results/{setting}/loss_curve.png')
        plt.show()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masked_preds = []
        masked_trues = []
        masks = []
        baseline_preds = []
        masked_baseline_preds = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = batch_x
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                pred_tensor = torch.tensor(pred)
                true_tensor = torch.tensor(true)
                masked_pred = torch.masked_select(pred_tensor, ~mask)
                masked_true = torch.masked_select(true_tensor, ~mask)
                baseline_pred = linear_interpolate_impute(batch_x.reshape(batch_x.shape[-2],batch_x.shape[-1]), mask.reshape(mask.shape[-2],mask.shape[-1]))
                masked_baseline_pred = torch.masked_select(baseline_pred, ~mask)

                preds.append(pred)
                trues.append(true)
                masked_preds.append(masked_pred.cpu().numpy())  # Convert before appending
                masked_trues.append(masked_true.cpu().numpy()) 
                masks.append(mask.cpu().numpy())
                baseline_preds.append(baseline_pred.cpu().numpy())
                masked_baseline_preds.append(masked_baseline_pred.cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        baseline_preds = np.array(baseline_preds)
        masked_preds = np.concatenate(masked_preds).reshape(-1)
        masked_trues = np.concatenate(masked_trues).reshape(-1)
        masked_baseline_preds = np.concatenate(masked_baseline_preds).reshape(-1)
        # print(masked_preds.shape, masked_trues.shape, masked_baseline_preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        print(preds.shape, trues.shape, masks.shape)
        preds = inverse_multi_tokens_array(preds, self.args.token_size, int(self.args.token_size/4), self.args.seq_len)
        trues = inverse_multi_tokens_array(trues, self.args.token_size, int(self.args.token_size/4), self.args.seq_len)
        masks = inverse_multi_tokens_array(masks, self.args.token_size, int(self.args.token_size/4), self.args.seq_len)
        print(preds.shape, trues.shape, masks.shape)
        baseline_preds = baseline_preds.reshape(-1, baseline_preds.shape[-2], baseline_preds.shape[-1])
        masked_preds = masked_preds.flatten()
        masked_trues = masked_trues.flatten()
        masked_baseline_preds = masked_baseline_preds.flatten()

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

         # Plot predictions vs ground truth for 3 random samples
        plt.figure(figsize=(15, 10))
        num_samples_to_plot = 3
        random_indices = np.random.choice(len(preds), size=num_samples_to_plot, replace=False)

        for i, idx in enumerate(random_indices):
            pred = preds[idx, :, 0]  # Use only the first feature (feature index 0)
            true = trues[idx, :, 0]  # Use only the first feature (feature index 0)
            mask = masks[idx, :, 0]  # Use only the first feature (feature index 0)
            baseline_pred = baseline_preds[idx, :, 0]  # Use only the first feature (feature index 0)
            mask = mask.astype(int)

            plt.subplot(num_samples_to_plot, 1, i + 1)
            plt.plot(range(len(true)), true, label="Ground Truth", color='blue')
            plt.plot(range(len(pred)), pred, label="Predicted Output", color='orange')
            plt.plot(range(len(mask)), mask, label="Mask", color='green')
            plt.plot(range(len(baseline_pred)), baseline_pred, label="Baseline Prediction", color='red')
    
            plt.xlabel("Timesteps")
            plt.ylabel("Values")
            plt.title(f"Predictions vs Ground Truth - Sample {i + 1} (Feature 0)")
            plt.legend()
            plt.grid(True)

        graph_path = os.path.join(folder_path, 'predictions_vs_ground_truth.png')
        plt.tight_layout()
        plt.savefig(graph_path)
        print(f"Saved comparison graph at {graph_path}")
        plt.close()

        mae, mse, rmse, mape, mspe = metric(masked_preds, masked_trues)
        # print(masked_baseline_preds.shape, masked_trues.shape)
        mae_baseline, mse_baseline, rmse_baseline, mape_baseline, mspe_baseline = metric(masked_baseline_preds, masked_trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('mse_baseline:{}, mae_baseline:{}'.format(mse_baseline, mae_baseline))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('baseline mse:{}, baseline mae:{}'.format(mse_baseline, mae_baseline))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return