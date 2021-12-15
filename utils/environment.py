def get_agent_types(env) -> [str]:
    types = []
    for agent in env.agents:
        if hasattr(agent, 'adversary'):
            if agent.adversary:
                types.append('adversary')
            else:
                types.append('ally')
        else:
            types.append('ally')
    return types
