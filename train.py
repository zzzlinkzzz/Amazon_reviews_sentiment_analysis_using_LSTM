from model import LSTMnet
from loader import make_dataloaders

from torch.nn import CrossEntropyLoss
import torch, os, sys

from tensorboardX import SummaryWriter
from optparse import OptionParser
from tqdm import tqdm
# =======================================================
# config
input_dim = 300
hidden_dim = 200
layer_dim = 1
output_dim = 2
num_workers = 2

dir_data = './dataset'
dir_checkpoint = './checkpoint/'
dir_log = dir_checkpoint + 'log/'

writer = SummaryWriter(dir_log)

torch.manual_seed(0)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# =======================================================
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='eta', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-f', '--data-dir', dest='data_dir', default=0.1,
                      type='str', help='dataset directory')
    (options, args) = parser.parse_args()
    return options
# =======================================================
def train_epoch(epoch, train_loader, test_loader, criterion, optimizer):
    net.train()
    epoch_loss = 0
    
    train_correct = 0
    total_train = 0
    test_correct = 0
    total_test = 0
    
    print("Train session:")
    for i, batch in enumerate(tqdm(train_loader)):
        #get data
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        #calculate
        outputs = net(texts)
        
        #learning
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        # predict
        predicted = torch.argmax(torch.softmax(outputs, dim=1),dim=1)
        
        # validate
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum()
    
    print("Test session:")
    for i, batch in enumerate(tqdm(test_loader)):
        #get data
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        #calculate
        outputs = net(texts)
        
        # predict
        predicted = torch.argmax(torch.softmax(outputs, dim=1),dim=1)
        
        # validate
        total_test += labels.size(0)
        test_correct += (predicted == labels).sum()
    
    # acuracy score
    train_accuracy = 100 * (train_correct / total_train)
    test_accuracy = 100 * (test_correct / total_test)
    print(f'Loss: {epoch_loss/i:.2f}  |  Train accuracy: {train_accuracy:.2f}%  |  Test accuracy: {test_accuracy:.2f}%')
    
    # logging
    writer.add_scalar('epoch_loss', epoch_loss/i, epoch+1)
    writer.add_scalar('train_accuracy', train_accuracy, epoch+1)
    writer.add_scalar('test_accuracy', test_accuracy, epoch+1)
    
    return (train_accuracy,test_accuracy)
# =======================================================
def train_net(train_loader, test_loader, net, epochs = 1, batch_size = 500, eta=0.1, save_cp=True):
    print(f'''
Training params:
    Epochs: {epochs}
    Batch_size: {batch_size}
    Learning rate: {eta}
    Training size: {len(train_loader.dataset)}
    Test size: {len(test_loader.dataset)}
    Device: {device}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=eta)
    criterion = CrossEntropyLoss()

    best_score = 0

    cp_list = [f for f in os.listdir(dir_checkpoint) if ".pth" in f]
    
    for epoch in range(len(cp_list),epochs):
        print(f'''{60*"="}\nStart epoch {epoch+1}''')
        cp_list = [f for f in os.listdir(dir_checkpoint) if ".pth" in f]
        
        if len(cp_list) != 0:
            trained_model = f'CP{len(cp_list)}.pth'
            net.load_state_dict(torch.load(dir_checkpoint+trained_model))
            net.eval()
            print('load '+ trained_model)

        score = train_epoch(epoch, train_loader, test_loader, criterion, optimizer)

        # scheduler.step()
        
        if save_cp and (score[0]>best_score):
            state_dict = net.state_dict()
            best_score = score[0]

        torch.save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
        print('Checkpoint {} saved !'.format(epoch + 1))
# =======================================================
if __name__ == "__main__":
    args = get_args()

    train_loader,test_loader = make_dataloaders('./dataset', args.data_dir, 'both', args.batch_size, num_workers)
    
    net = LSTMnet(input_dim, hidden_dim, layer_dim, output_dim, device).to(device)

    try:
        train_net(train_loader, test_loader, net, epochs = args.epochs, batch_size = args.batch_size, eta = args.eta)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
# =======================================================