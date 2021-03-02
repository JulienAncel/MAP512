if __name__ == "__main__":
    from src.test import multimeasure
    #multimeasure.histogram_test(-1.5, 0.5, int(1e6), 1000)
    #multimeasure.limite_law(-1.5, 0.5, int(1e4), int(1e3), 1000)
    multimeasure.limite_law(-1.5, 0.5, int(1e4), int(1e3), 0)
    #multimeasure.potential()
    #OU.single_curve([0, 1, 2])
    #OU.heatmap([1, 2])
    #OU.tcl(M=50)
