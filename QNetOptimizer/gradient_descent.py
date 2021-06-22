import pennylane as qml

def gradient_descent(cost, init_settings, num_steps=150, step_size=0.1, sample_width=25, verbose=True):
    opt = qml.GradientDescentOptimizer(stepsize=step_size)    

    settings = init_settings
    scores = []
    samples = []
    settings_history = []

    # performing gradient descent
    for i in range(num_steps):
        settings = opt.step(cost, settings)
        settings_history.append(settings)
        
        if i%sample_width == 0:
            score = -(cost(settings))
            scores.append(score)
            samples.append(i)

            if verbose:
                print("iteration : ",i, ", score : ", score)
                print("settings :\n", settings, "\n")
    
    opt_score = -(cost(settings))
    
    scores.append(opt_score)
    samples.append(num_steps-1)
    
    return {
        "opt_score": opt_score,
        "opt_settings": settings,
        "scores": scores,
        "samples": samples,
        "settings_history": settings_history
    }
