import pennylane as qml

def gradient_descent(cost, init_settings, num_steps=150, step_size=0.1, sample_width=25, grad_fn=None):
    opt = qml.GradientDescentOptimizer(stepsize=step_size)    

    settings = init_settings
    scores = []
    samples = []
    settings_history = []

    # performing gradient descent
    for i in range(num_steps):
        if i%sample_width == 0:
            score = -(cost(settings))
            scores.append(score)
            samples.append(i)

            print("iteration : ",i, ", score : ", score)
            print("settings :\n", settings, "\n")
        
        settings = opt.step(cost, settings, grad_fn=grad_fn)
        settings_history.append(settings)
    
    final_score = -(cost(settings))
    
    scores.append(final_score)
    samples.append(num_steps-1)
    
    return (final_score, settings, scores, samples, settings_history)