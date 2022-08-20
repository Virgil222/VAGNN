import world
import utils
from world import cprint
import torch
import time
import Procedure

# ==============================
#utils.set_seed(world.seed)
#print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
#Recmodel.load_state_dict(torch.load('checkpoints/lgn-tiktok_feed-2-6441.model'))

Neg_k = 1



best_hr, best_ndcg = 0, 0
best_model = Recmodel
best_epoch = 0
count, epoch = 0, 0

while count < 8:
    start = time.time()
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
    cprint("[valid]")
    res = Procedure.Test(dataset, Recmodel, 'valid', world.config['multicore'])
    hr1, ndcg1 = res['recall'][0], res['ndcg'][0]
    hr2, ndcg2 = res['recall'][2], res['ndcg'][2]
    print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
    if hr1 > best_hr :
        best_epoch = epoch
        best_model = Recmodel
        count = 0
        best_hr, best_ndcg = hr1, ndcg1

    else:
        # 小于三次退出训练
        count += 1
    epoch += 1

model_dir = weight_file + str(best_epoch) + '.model'
print(model_dir)
torch.save(best_model.state_dict(), model_dir)

print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f} in invalid data".format(
    best_epoch, best_hr, best_ndcg))

# test
model = best_model
cprint("[test]")
res = Procedure.Test(dataset, Recmodel, 'test', world.config['multicore'])

