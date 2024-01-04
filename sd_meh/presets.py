# fmt: off
BLOCK_WEIGHTS_PRESETS = {
    "GRAD_V": [0, 1, 0.9166666667, 0.8333333333, 0.75, 0.6666666667, 0.5833333333, 0.5, 0.4166666667, 0.3333333333, 0.25, 0.1666666667, 0.0833333333, 0, 0.0833333333, 0.1666666667, 0.25, 0.3333333333, 0.4166666667, 0.5, 0.5833333333, 0.6666666667, 0.75, 0.8333333333, 0.9166666667, 1.0],
    "GRAD_A": [0, 0, 0.0833333333, 0.1666666667, 0.25, 0.3333333333, 0.4166666667, 0.5, 0.5833333333, 0.6666666667, 0.75, 0.8333333333, 0.9166666667, 1.0, 0.9166666667, 0.8333333333, 0.75, 0.6666666667, 0.5833333333, 0.5, 0.4166666667, 0.3333333333, 0.25, 0.1666666667, 0.0833333333, 0],
    "FLAT_25": [0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    "FLAT_75": [0, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
    "WRAP08": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    "WRAP12": [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "WRAP14": [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    "WRAP16": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    "MID12_50": [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
    "OUT07": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    "OUT12": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "OUT12_5": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "RING08_SOFT": [0, 0, 0, 0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0, 0, 0],
    "RING08_5": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "RING10_5": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "RING10_3": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "SMOOTHSTEP": [0, 0, 0.00506365740740741, 0.0196759259259259, 0.04296875, 0.0740740740740741, 0.112123842592593, 0.15625, 0.205584490740741, 0.259259259259259, 0.31640625, 0.376157407407407, 0.437644675925926, 0.5, 0.562355324074074, 0.623842592592592, 0.68359375, 0.740740740740741, 0.794415509259259, 0.84375, 0.887876157407408, 0.925925925925926, 0.95703125, 0.980324074074074, 0.994936342592593, 1],
    "REVERSE_SMOOTHSTEP": [0, 1, 0.994936342592593, 0.980324074074074, 0.95703125, 0.925925925925926, 0.887876157407407, 0.84375, 0.794415509259259, 0.740740740740741, 0.68359375, 0.623842592592593, 0.562355324074074, 0.5, 0.437644675925926, 0.376157407407408, 0.31640625, 0.259259259259259, 0.205584490740741, 0.15625, 0.112123842592592, 0.0740740740740742, 0.0429687499999996, 0.0196759259259258, 0.00506365740740744, 0],
    "2SMOOTHSTEP": [0, 0, 0.0101273148148148, 0.0393518518518519, 0.0859375, 0.148148148148148, 0.224247685185185, 0.3125, 0.411168981481482, 0.518518518518519, 0.6328125, 0.752314814814815, 0.875289351851852, 1.0, 0.875289351851852, 0.752314814814815, 0.6328125, 0.518518518518519, 0.411168981481481, 0.3125, 0.224247685185184, 0.148148148148148, 0.0859375, 0.0393518518518512, 0.0101273148148153, 0],
    "2R_SMOOTHSTEP": [0, 1, 0.989872685185185, 0.960648148148148, 0.9140625, 0.851851851851852, 0.775752314814815, 0.6875, 0.588831018518519, 0.481481481481481, 0.3671875, 0.247685185185185, 0.124710648148148, 0.0, 0.124710648148148, 0.247685185185185, 0.3671875, 0.481481481481481, 0.588831018518519, 0.6875, 0.775752314814816, 0.851851851851852, 0.9140625, 0.960648148148149, 0.989872685185185, 1],
    "3SMOOTHSTEP": [0, 0, 0.0151909722222222, 0.0590277777777778, 0.12890625, 0.222222222222222, 0.336371527777778, 0.46875, 0.616753472222222, 0.777777777777778, 0.94921875, 0.871527777777778, 0.687065972222222, 0.5, 0.312934027777778, 0.128472222222222, 0.0507812500000004, 0.222222222222222, 0.383246527777778, 0.53125, 0.663628472222223, 0.777777777777778, 0.87109375, 0.940972222222222, 0.984809027777777, 1],
    "3R_SMOOTHSTEP": [0, 1, 0.984809027777778, 0.940972222222222, 0.87109375, 0.777777777777778, 0.663628472222222, 0.53125, 0.383246527777778, 0.222222222222222, 0.05078125, 0.128472222222222, 0.312934027777778, 0.5, 0.687065972222222, 0.871527777777778, 0.94921875, 0.777777777777778, 0.616753472222222, 0.46875, 0.336371527777777, 0.222222222222222, 0.12890625, 0.0590277777777777, 0.0151909722222232, 0],
    "4SMOOTHSTEP": [0, 0, 0.0202546296296296, 0.0787037037037037, 0.171875, 0.296296296296296, 0.44849537037037, 0.625, 0.822337962962963, 0.962962962962963, 0.734375, 0.49537037037037, 0.249421296296296, 0.0, 0.249421296296296, 0.495370370370371, 0.734375000000001, 0.962962962962963, 0.822337962962962, 0.625, 0.448495370370369, 0.296296296296297, 0.171875, 0.0787037037037024, 0.0202546296296307, 0],
    "4R_SMOOTHSTEP": [0, 1, 0.97974537037037, 0.921296296296296, 0.828125, 0.703703703703704, 0.55150462962963, 0.375, 0.177662037037037, 0.0370370370370372, 0.265625, 0.50462962962963, 0.750578703703704, 1.0, 0.750578703703704, 0.504629629629629, 0.265624999999999, 0.0370370370370372, 0.177662037037038, 0.375, 0.551504629629631, 0.703703703703703, 0.828125, 0.921296296296298, 0.979745370370369, 1],
    "HALF_SMOOTHSTEP": [0, 0, 0.0196759259259259, 0.0740740740740741, 0.15625, 0.259259259259259, 0.376157407407407, 0.5, 0.623842592592593, 0.740740740740741, 0.84375, 0.925925925925926, 0.980324074074074, 1.0, 0.980324074074074, 0.925925925925926, 0.84375, 0.740740740740741, 0.623842592592593, 0.5, 0.376157407407407, 0.259259259259259, 0.15625, 0.0740740740740741, 0.0196759259259259, 0],
    "HALF_R_SMOOTHSTEP": [0, 1, 0.980324074074074, 0.925925925925926, 0.84375, 0.740740740740741, 0.623842592592593, 0.5, 0.376157407407407, 0.259259259259259, 0.15625, 0.0740740740740742, 0.0196759259259256, 0.0, 0.0196759259259256, 0.0740740740740742, 0.15625, 0.259259259259259, 0.376157407407407, 0.5, 0.623842592592593, 0.740740740740741, 0.84375, 0.925925925925926, 0.980324074074074, 1],
    "ONE_THIRD_SMOOTHSTEP": [0, 0, 0.04296875, 0.15625, 0.31640625, 0.5, 0.68359375, 0.84375, 0.95703125, 1.0, 0.95703125, 0.84375, 0.68359375, 0.5, 0.31640625, 0.15625, 0.04296875, 0.0, 0.04296875, 0.15625, 0.31640625, 0.5, 0.68359375, 0.84375, 0.95703125, 1],
    "ONE_THIRD_R_SMOOTHSTEP": [0, 1, 0.95703125, 0.84375, 0.68359375, 0.5, 0.31640625, 0.15625, 0.04296875, 0.0, 0.04296875, 0.15625, 0.31640625, 0.5, 0.68359375, 0.84375, 0.95703125, 1.0, 0.95703125, 0.84375, 0.68359375, 0.5, 0.31640625, 0.15625, 0.04296875, 0],
    "ONE_FOURTH_SMOOTHSTEP": [0, 0, 0.0740740740740741, 0.259259259259259, 0.5, 0.740740740740741, 0.925925925925926, 1.0, 0.925925925925926, 0.740740740740741, 0.5, 0.259259259259259, 0.0740740740740741, 0.0, 0.0740740740740741, 0.259259259259259, 0.5, 0.740740740740741, 0.925925925925926, 1.0, 0.925925925925926, 0.740740740740741, 0.5, 0.259259259259259, 0.0740740740740741, 0],
    "ONE_FOURTH_R_SMOOTHSTEP": [0, 1, 0.925925925925926, 0.740740740740741, 0.5, 0.259259259259259, 0.0740740740740742, 0.0, 0.0740740740740742, 0.259259259259259, 0.5, 0.740740740740741, 0.925925925925926, 1.0, 0.925925925925926, 0.740740740740741, 0.5, 0.259259259259259, 0.0740740740740742, 0.0, 0.0740740740740742, 0.259259259259259, 0.5, 0.740740740740741, 0.925925925925926, 1],
    "COSINE": [0, 1, 0.995722430686905, 0.982962913144534, 0.961939766255643, 0.933012701892219, 0.896676670145617, 0.853553390593274, 0.80438071450436, 0.75, 0.691341716182545, 0.62940952255126, 0.565263096110026, 0.5, 0.434736903889974, 0.37059047744874, 0.308658283817455, 0.25, 0.195619285495639, 0.146446609406726, 0.103323329854382, 0.0669872981077805, 0.0380602337443566, 0.0170370868554658, 0.00427756931309475, 0],
    "REVERSE_COSINE": [0, 0, 0.00427756931309475, 0.0170370868554659, 0.0380602337443566, 0.0669872981077808, 0.103323329854383, 0.146446609406726, 0.19561928549564, 0.25, 0.308658283817455, 0.37059047744874, 0.434736903889974, 0.5, 0.565263096110026, 0.62940952255126, 0.691341716182545, 0.75, 0.804380714504361, 0.853553390593274, 0.896676670145618, 0.933012701892219, 0.961939766255643, 0.982962913144534, 0.995722430686905, 1],
    "TRUE_CUBIC_HERMITE": [0, 0, 0.199031876929012, 0.325761959876543, 0.424641927083333, 0.498456790123457, 0.549991560570988, 0.58203125, 0.597360869984568, 0.598765432098765, 0.589029947916667, 0.570939429012346, 0.547278886959876, 0.520833333333333, 0.49438777970679, 0.470727237654321, 0.45263671875, 0.442901234567901, 0.444305796682099, 0.459635416666667, 0.491675106095678, 0.543209876543211, 0.617024739583333, 0.715904706790124, 0.842634789737655, 1],
    "TRUE_REVERSE_CUBIC_HERMITE": [0, 1, 0.800968123070988, 0.674238040123457, 0.575358072916667, 0.501543209876543, 0.450008439429012, 0.41796875, 0.402639130015432, 0.401234567901235, 0.410970052083333, 0.429060570987654, 0.452721113040124, 0.479166666666667, 0.50561222029321, 0.529272762345679, 0.54736328125, 0.557098765432099, 0.555694203317901, 0.540364583333333, 0.508324893904322, 0.456790123456789, 0.382975260416667, 0.284095293209876, 0.157365210262345, 0],
    "FAKE_CUBIC_HERMITE": [0, 0, 0.157576195987654, 0.28491512345679, 0.384765625, 0.459876543209877, 0.512996720679012, 0.546875, 0.564260223765432, 0.567901234567901, 0.560546875, 0.544945987654321, 0.523847415123457, 0.5, 0.476152584876543, 0.455054012345679, 0.439453125, 0.432098765432099, 0.435739776234568, 0.453125, 0.487003279320987, 0.540123456790124, 0.615234375, 0.71508487654321, 0.842423804012347, 1],
    "FAKE_REVERSE_CUBIC_HERMITE": [0, 1, 0.842423804012346, 0.71508487654321, 0.615234375, 0.540123456790123, 0.487003279320988, 0.453125, 0.435739776234568, 0.432098765432099, 0.439453125, 0.455054012345679, 0.476152584876543, 0.5, 0.523847415123457, 0.544945987654321, 0.560546875, 0.567901234567901, 0.564260223765432, 0.546875, 0.512996720679013, 0.459876543209876, 0.384765625, 0.28491512345679, 0.157576195987653, 0],
    "ALL_A": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ALL_B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
# fmt: on
