from experiment import EffectDrivenVisualServiceSelectionExperiment


def main():
    """ unit of distance is Meter """
    experiment = EffectDrivenVisualServiceSelectionExperiment(num_device=100,
                                                              width=10,
                                                              height=10,
                                                              depth=3,
                                                              max_speed=1,
                                                              observation_range=3,
                                                              num_episode=1000,
                                                              num_step=100,
                                                              memory_size=100,
                                                              batch_size=100,
                                                              learning_rate=0.00000001,
                                                              discount_factor=1.,
                                                              agent="DRRN")
    experiment.run()


if __name__ == '__main__':
    main()
