import numpy as np
import ROOT
def loadAllFFHist(basedirname):
   pass  

def loadFlatfieldHist(fname):
    NXaxis = 765  # GC: NXaxis and NYaxis have to match the size of the .fits files. All .fits are 760x510 pixels .
    NYaxis = 510
    fFlatfield = ROOT.TFile(fname)

    hFlatfield = ROOT.TH2F()

               #print(hPicture)
    hFlatfield = fFlatfield.Get("hPicture");

    hFlatDistr = ROOT.TH1F("hFlatDistr","hFlatDistr", 65535, 0, 65535);

    for jj in range(NXaxis):
        for kk in range(NYaxis):
            hFlatDistr.Fill(hFlatfield.GetBinContent(jj, kk))

    hFlatfield.Scale(1./hFlatDistr.GetMean())
    hFlatfield.Smooth(1)
    data_arr = np.zeros((NYaxis,NXaxis))
    for jj in range(NXaxis):
        for kk in range(NYaxis):
            data_arr[kk,jj] = hFlatfield.GetBinContent(jj+1, kk+1)
    return data_arr


