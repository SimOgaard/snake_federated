# Torch modules
from torch import no_grad
from torch import div as torch_div

def agregate(agents: list, *arg) -> None:
    '''
    Agregates between each agent in agents and each model in agent
    '''
    def agregate_model(models: list):
        '''
        Returns agregated median of all models
        '''
        with no_grad():
            model_average = dict(models[0].named_parameters())

            for key in model_average.keys():
                for i in range(1, len(models)):
                    model_average[key].data += dict(models[i].named_parameters())[key].data

                model_average[key].data = torch_div(model_average[key].data, len(models))

        return model_average

    # agregate both models
    agregated_qnetwork_local = agregate_model([agent.qnetwork_local for agent in agents])
    agregated_qnetwork_target = agregate_model([agent.qnetwork_target for agent in agents])

    # apply a copied version of the returning agregated model to each agent
    for agent in agents + list(arg):
        agent.qnetwork_local.load_state_dict(agregated_qnetwork_local)
        agent.qnetwork_target.load_state_dict(agregated_qnetwork_target)