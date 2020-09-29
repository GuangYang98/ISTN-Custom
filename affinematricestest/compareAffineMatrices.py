import pickle
import numpy as np
from io import StringIO

matricesTuples = []
matricesTuples.append((pickle.load(open('affineMatrices_refined_e.dat', "rb")), 'affineMatrices_refined_e'))
matricesTuples.append((pickle.load(open('affineMatrices_refined_i.dat', "rb")), 'affineMatrices_refined_i'))
matricesTuples.append((pickle.load(open('affineMatrices_refined_s.dat', "rb")), 'affineMatrices_refined_s'))
matricesTuples.append((pickle.load(open('affineMatrices_refined_u.dat', "rb")), 'affineMatrices_refined_u'))

matricesTuples.append((pickle.load(open('affineMatrices_warped_e.dat', "rb")), 'affineMatrices_warped_e'))
matricesTuples.append((pickle.load(open('affineMatrices_warped_i.dat', "rb")), 'affineMatrices_warped_i'))
matricesTuples.append((pickle.load(open('affineMatrices_warped_s.dat', "rb")), 'affineMatrices_warped_s'))
matricesTuples.append((pickle.load(open('affineMatrices_warped_u.dat', "rb")), 'affineMatrices_warped_u'))

originalAffineMatrices = pickle.load(open('originalAffineMatrices.dat', "rb"))


with open("AffineMatrixResults.txt", "w") as f:
    for (matrices,name) in matricesTuples:
        with StringIO() as out:
            out.write("Name:\n")
            out.write(name + '\n')
            

            As = originalAffineMatrices
            Bs = matrices

            MSEs = []
            for i in range(60):
                A = As[i]
                B = Bs[i]

                mse = (np.square(A - B)).mean(axis=None)

                MSEs.append(mse)
            
            mseAverage = np.mean(MSEs)
            mseVariance = np.var(MSEs)

            out.write("Average MSE:\n")
            out.write(str(mseAverage) + '\n')
            out.write("Variance MSE:\n")
            out.write(str(mseVariance) + '\n')
            out.write('\n')
            f.write(out.getvalue())





