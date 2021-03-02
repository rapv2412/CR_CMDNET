

def norm_replace(imcloudy, impred, imrepl == False):

    sizexy = imcloudy[0][0].size() #Get size
    sizex, sizey = sizexy[0] , sizexy[1]

    imnorm = impred.clone()
    imrepl = imcloudy.clone()

    for i in range(len(imcloudy)):
        for j in range(len(imcloudy[0])):

            imc = imcloudy[i][j]
            imp = impred[i][j]
            
            #Assume Cloudy pixels have value = 1
            #Added !=1 to exclude cloudy pixels from mean/std calcs

            #Pf is from cloud contaminated image
            Pf_mu = imc[imc!=1].mean() 
            Pf_sig = imc[imc!=1].std()

            #Pe is from estimated/predicted image
            Pe_mu = imp.mean() 
            Pe_sig = imp.std()

            #only do normalization 
            for x in range(sizex):
                for y in range(sizey):
                    imnorm[i][j][x][y] = (imp[x][y] - Pe_mu)*(Pf_sig/Pe_sig)+Pf_mu
                    if imc[x][y] == 1:
                        imrepl[i][j][x][y] = imnorm[i][j][x][y]

    if imrepl == False:
        return imnorm
    else:
        return imnorm, imrepl

        