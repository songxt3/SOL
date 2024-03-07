import torch
from transformers import BertModel
from util import get_metrics, compute_metrics, prepare_inputs
from new_dataloader import get_dataloader
from meta_optimizer import Solo

class BERTClassifierModel(torch.nn.Module):
    def __init__(self, model_checkpoint, num_labels, task=None):
        super(BERTClassifierModel, self).__init__()
        self.task = task
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.linear = torch.nn.Linear(768, num_labels)
        if task == "stsb":
            self.loss = torch.nn.MSELoss()
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.metric, self.metric_1 = get_metrics(task)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear_output = self.linear(bert_output['last_hidden_state'][:, 0, :].view(-1, 768))
        if self.task == "stsb":
            linear_output = torch.clip((self.sigmoid(linear_output) * 5.5), min=0.0, max=5.0)
        output = {"hidden_layer": bert_output, "logits": linear_output}
        return output

    def compute_metrics(self, predictions, references):
        metric, metric_1 = None, None
        if self.metric is not None:
            metric = compute_metrics(predictions=predictions, references=references, metric=self.metric)
        if self.metric_1 is not None:
            metric_1 = compute_metrics(predictions=predictions, references=references, metric=self.metric_1)
        return metric, metric_1


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate(net, val_iter):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    iterator = iter(val_iter)
    len_iterator = len(val_iter)
    with torch.no_grad():
        for step in range(len_iterator):
            inputs = next(iterator)
            inputs = prepare_inputs(inputs, device)
            if 'labels' in inputs:
                labels = inputs.pop('labels')
            outputs = net(**inputs)

            logits = outputs['logits']
            acc, acc1 = net.compute_metrics(logits, labels)
            mean_acc += acc['accuracy']
            # mean_acc += acc['matthews_correlation']
            count += 1

    return mean_acc / count, mean_loss / count


def train(net, optimizer, train_iter, val_iter):
    best_acc = 0
    for ep in range(3):
        train_iter_len = len(train_iter)
        iterator = iter(train_iter)
        for step in range(train_iter_len):
            net.train()
            # Clear gradients
            net.zero_grad()
            # Converting these to cuda tensors
            inputs = next(iterator)
            inputs = prepare_inputs(inputs, device)
            if "labels" in inputs:
                labels = inputs.pop('labels')

            # Obtaining the logits from the model
            outputs = net(**inputs)
            logits = outputs['logits']

            # Computing loss
            loss = net.loss(logits, labels)

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            optimizer.step()
            # scheduler.step()

            if step % 100 == 0:
                acc, acc_1 = net.compute_metrics(predictions=logits, references=labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(step, ep, loss.item(), acc))

        val_acc, val_loss = evaluate(net, val_iter)
        net.train()
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
    print("Best accuracy:", best_acc.item() * 100)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = './bert-base-uncased'
    task = "rte"
    batch_size = 64
    lr = 5e-5

    # Load DataLoader
    print(f"\nLoading data...")
    train_epoch_iterator = get_dataloader(task, model_checkpoint, "train", batch_size=batch_size)
    eval_epoch_iterator = get_dataloader(task, model_checkpoint, "validation", batch_size=batch_size)

    # Load Pre-trained Model
    print(f"\nLoading pre-trained BERT model \"{model_checkpoint}\"")
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    net = BERTClassifierModel(model_checkpoint, num_labels=num_labels, task=task).to(device)

    # expr = '0.739265*g + 0.561438*m + 0.586163*v - 2.04334e-6*(-g + 0.981843*v)**2 + 0.209439501166344*(0.0761669*g + 0.0690125*m + 0.0778506*v)*(0.0796514*g + 0.060741*m + 0.0770472*v) - 0.474898769050401*exp(-0.193474*g - 0.216027*m - 0.18707*v) + 0.0810726*sign(m) + 0.183848364740174 + 0.570163/(exp(-5.09114*g - 3.85242*m - 2.9287*v) + 1)'
    expr = '0.0117786*1 - 0.000582272*2 + 0.0642267*ad - 0.0278209*ag + 0.0609039*g + 0.00792243*g2 + 0.00654308*g3 + 0.00447625*m + 0.0332765*rs + 0.0343053*sg + 0.0977111*sm + 0.00549125*v + 0.000788384*(-0.727733*1 + 0.320973*2 + 0.507005*ad + ag - 0.0753359*g - 0.263634*m + 0.125063*rs - 0.77929*sg - 0.587381*sm + 0.0449483*v)**2 + 0.847640991210938*(0.00664244*1 - 0.00723596*2 + 0.135008*ad + 0.0247901*ag + 0.0362407*g - 0.00450618*g2 + 0.00825409*g3 + 0.067325*m + 0.0941577*rs + 0.102553*sg + 0.158247*sm + 0.0124416*v)*(0.195097*1 + 0.088762*2 - 0.063345*ad - 0.223571*ag - 0.0145187*g - 0.00646708*g2 + 0.00448548*g3 + 0.0140364*m + 0.029944*rs - 0.00861674*sg - 0.0135936*sm + 0.130794*v) - 0.402384847401704*exp(-0.0156423*1 + 0.0129938*2 - 0.0216611*ad - 0.059775*ag + 0.0115238*g - 0.0469424*g2 - 0.116046*m - 0.124361*rs - 0.0407348*sg - 0.132727*sm - 0.0596081*v) - 0.202378*sign(0.0015269*1 - 0.0378086*g - 0.013156*m - 0.00900533*sm) + 1.72200176119718 - 1.18457/(exp(-1.31462*2 + 1.77828*ad + 0.241781*ag + 1.26038*g + 0.740951*g2 + 0.160378*g3 + 1.83776*m + 1.602*rs + 2.13443*sg + 0.905846*sm + 0.681621*v) + 1)'
    # expr = '0.0598765*1 + 0.0389852*2 + 0.0306917*ad - 0.0787246*ag + 0.0142268*g + 0.0175024*g2 + 0.0229455*g3 + 0.0541093*m + 0.0463721*rs + 0.0218315*sg + 0.0606016*sm + 0.0556724*v + 0.0590855*(-0.510783*1 - 2 - 0.406711*ad + 0.666904*ag - 0.730698*g - 0.459349*g2 - 0.569153*g3 - 0.918479*m - 0.694792*rs - 0.388724*sg - 0.643419*sm - 0.765329*v)**2 - 0.437239348888397*(-0.211906*1 - 0.176794*2 - 0.0838447*ad + 0.0912917*ag - 0.0475501*g - 0.053051*g2 - 0.0358499*g3 - 0.0545907*m - 0.129909*rs - 0.151686*sg - 0.0205031*sm - 0.160727*v)*(0.167059*1 + 0.0457043*2 + 0.251547*ad - 0.0912737*ag + 0.144436*g + 0.0430009*g2 + 0.0316128*g3 + 0.0691217*m + 0.19881*rs + 0.209708*sg + 0.19168*sm + 0.0521466*v) + 0.60133377700774*exp(0.13269*1 + 0.165798*2 + 0.1833*ad - 0.0670627*ag + 0.103352*g + 0.0902946*g2 + 0.0871466*g3 + 0.238537*m + 0.0520296*rs + 0.128674*sg + 0.149263*sm + 0.171151*v) + 0.784951*sign(-0.00114344*m + 0.00234473*v) - 0.56754613533346 - 1.15277/(exp(1.44099*1 + 0.790812*2 + 0.69248*ad - 1.19276*ag + 3.00986*g + 0.178428*g2 + 0.641186*g3 + 2.56261*m + 2.6058*rs + 3.47013*sg + 0.249456*sm + 0.970683*v) + 1)'
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    optimizer = Solo(net.parameters(), lr=lr, expr=expr)

    train(net, optimizer, train_epoch_iterator, eval_epoch_iterator)
