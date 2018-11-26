from experiment import EffectDrivenVisualServiceSelectionExperiment


def main():
    """ unit of distance is Meter """
    experiment = EffectDrivenVisualServiceSelectionExperiment(num_device=100,
                                                              width=10,
                                                              height=10,
                                                              depth=10,
                                                              max_speed=1,
                                                              observation_range=3,
                                                              num_episode=1000,
                                                              num_step=100,
                                                              memory_size=10,
                                                              batch_size=10)
    experiment.run()


if __name__ == '__main__':
    main()
