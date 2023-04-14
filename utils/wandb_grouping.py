import wandb
from exp_recommendation.mykey import wandb_login_key, wandb_project_name, wandb_entity_name

# if __name__ == '__main__':
#     wandb.login(key=wandb_login_key)
#     api = wandb.Api()
#     for r in api.runs("{}/{}".format(wandb_entity_name, wandb_project_name), filters={"group": "<null>"}):
#         r.group = "exp6_"
#         r.update()

if __name__ == '__main__':
    wandb.login(key=wandb_login_key)
    api = wandb.Api()

    # r = api.run("yuelin301/IND+MARL/3ssjhozm")

    # for r in api.runs("{}/{}".format(wandb_entity_name, wandb_project_name), filters={"tag": "final"}):
    for r in api.runs("<yuelin301/IND+MARL>", filters={"tag": "final"}):
        r.group = "exp6_"
        r.update()
