from torch.nn import CrossEntropyLoss
from torch import optim, load, save, cuda

from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")
embedding = GloVe(name = '840B', dim = 300)

from model import LSTMModel
from loader import make_dataloaders
from validation import eval_net_loader

from tqdm import tqdm
from optparse import OptionParser
import os,sys
# ============================================================
accumulation_steps = 16
gamma=0.97
dir_data = './dataset'
dir_checkpoint = './checkpoint/'
num_workers =  1
is_culmulated = True
# ============================================================
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='eta', default=0.1,
                      type='float', help='learning rate')
    (options, args) = parser.parse_args()
    return options
# ============================================================
def train_epoch(train_loader, criterion, optimizer):
    net.train()
    epoch_loss = 0
    batch_num = len(train_loader)

    for i, batch in enumerate(tqdm(train_loader)):
        imgs = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = net(imgs)
        # probs = torch.softmax(outputs, dim=1)

        loss = criterion(outputs, masks)
        epoch_loss += loss.detach().to('cpu').item()
        loss.backward()

        if is_culmulated:
            loss /= accumulation_steps
            if ((i+1) % accumulation_steps == 0) or ((i+1) == batch_num):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
    
    print(f'Loss: {epoch_loss/i:.2f}')
# ============================================================
def validate_epoch(val_loader, device):
    classIOU, meanIOU = eval_net_loader(net, val_loader, 5, device)
    print('Class IoU:', ','.join(f'{x:.3f}' for x in classIOU))
    print(f'Mean IoU: {meanIOU:.3f}') #  |  
    return meanIOU
# ============================================================
def train_net(train_loader, val_loader, net, device, epochs = 1, batch_size = 2, eta=0.1, save_cp=True):
    print(f'''
Training params:
    Epochs: {epochs}
    Batch_size: {batch_size}
    Learning rate: {eta}
    Training size: {len(train_loader.dataset)}
    Validation size: {len(val_loader.dataset)}
    Device: {device}
    accumulation steps: {accumulation_steps}
    Decreasing Learning rate by {round((1-gamma),2)*100}% per epoch
    ''')

    optimizer = optim.Adam(net.parameters(),  lr= eta, weight_decay=0.1)
    criterion = CrossEntropyLoss()

    best_precision = 0

    cp_list = [f for f in os.listdir(dir_checkpoint) if ".pth" in f]
    
    for epoch in range(len(cp_list),epochs):
        print(f'''============================================================\nStart epoch {epoch+1}\nTraining session''')
        cp_list = [f for f in os.listdir(dir_checkpoint) if ".pth" in f]
        
        if len(cp_list) != 0:
            trained_model = f'CP{len(cp_list)}.pth'
            # net = Unet_model(in_channels=3 ,n_classes=5)
            net.load_state_dict(load(dir_checkpoint+trained_model))
            net.eval()
            print('load '+ trained_model)

        train_epoch(train_loader, criterion, optimizer)
        print("Validation session")
        precision = validate_epoch(val_loader, device)
        # scheduler.step()
        
        if save_cp and (precision>best_precision):
            state_dict = net.state_dict()
            best_precision = precision

        save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
        print('Checkpoint {} saved !'.format(epoch + 1))
# ============================================================
if __name__ == "__main__":
    device = ('cuda' if cuda.is_available() else 'cpu')
    
    args = get_args()
    
    train_loader,test_loader = make_dataloaders(dir_data, tokenizer, embedding, 'both', args.batch_size, num_workers)
    net = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    net.to(device)

    try:
        train_net(train_loader, val_loader, net, device, epochs=args.epochs, batch_size=args.batch_size, eta=args.eta)
    except KeyboardInterrupt:
        save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)