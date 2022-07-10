import sys, os, glob
from copy import deepcopy
import time

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import modeller
from modeller import *
from modeller.scripts import complete_pdb

import MDAnalysis as mda

import biobox as bb

import warnings
warnings.filterwarnings("ignore")

#edit path as required for your computer (or remove, if you installed molearn via conda-forge)
sys.path.insert(0, "C:\\Users\\xdzl45\\workspace\\molearn\\src")
from molearn import load_data, Auto_potential, Autoencoder, ResidualBlock

from ipywidgets import HBox, VBox, Layout
from ipywidgets import widgets
from tkinter import Tk, filedialog
import plotly.graph_objects as go
import nglview as nv



class MolearnAnalysis(object):
    
    def __init__(self, networkfile, infile, m=2.0, latent_z=2, r=2, atoms = ["CA", "C", "N", "CB", "O"]):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        training_set, meanval, stdval, atom_names, mol, test0, test1 = load_data(infile,
                                                                                 atoms = atoms,
                                                                                 device = self.device)
      
        # set residues names with protonated histidines back to generic HIS name (needed by DOPE score function)
        testH = mol.data["resname"].values
        testH[testH == "HIE"] = "HIS"
        testH[testH == "HID"] = "HIS"
        mol.data["resname"] = testH
        
        # create an MDAnalysis instance of input protein
        mol.write_pdb("tmp.pdb")
        self.mymol = mda.Universe('tmp.pdb')

        self.training_set = training_set
        self.meanval = meanval
        self.stdval = stdval
        self.mol = mol
        self.atoms = atoms
        
        checkpoint = torch.load(networkfile, map_location=self.device)
        self.network = Autoencoder(m=m, latent_z=latent_z, r=r).to(self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])

        for modulelist in [self.network.encoder, self.network.decoder]:
            for layer in modulelist:
                if type(layer)==torch.nn.BatchNorm1d:
                    layer.momentum=1.0
                elif type(layer)==ResidualBlock:
                    for rlayer in layer.conv_block:
                        if type(rlayer)==torch.nn.BatchNorm1d:
                            rlayer.momentum=1.0

        with torch.no_grad():
            self.network.decode(self.network.encode(self.training_set.float()))

        self.network.eval()
        
        with torch.no_grad():
            z = self.network.encode(self.training_set.float())
            self.z_training = z.data.cpu().numpy()[:, :, 0]
        
        os.remove("rmsd_matrix.npy")
    
    
    def load_test(self, infile):

        self.test_set, _, _, _, _, _, _ = load_data(infile, atoms = self.atoms, device=self.device)
        if self.test_set.shape[2] != self.training_set.shape[2]:
            raise Exception(f'number of d.o.f. differs: training set has {self.training_set.shape[2]}, test set has {test_set.shape[2]}')

        with torch.no_grad():
            z = self.network.encode(self.test_set.float())
            self.z_test = z.data.cpu().numpy()[:, :, 0]
            
        return self.test_set, self.z_test


    def get_error(self, dataset="", align=False):
        '''
        Calculate the reconstruction error of a dataset encoded and decoded by a trained neural network
        '''

        if dataset == "":
            dataset = self.training_set

        z = self.network.encode(dataset.float())
        decoded = self.network.decode(z)[:,:,:dataset.shape[2]]

        err = []
        for i in range(dataset.shape[0]):

            crd_ref = dataset[i].permute(1,0).unsqueeze(0).data.cpu().numpy()*self.stdval + self.meanval
            crd_mdl = decoded[i].permute(1,0).unsqueeze(0).data.cpu().numpy()[:, :dataset.shape[2]]*self.stdval + self.meanval #clip the padding of models  

            if align: # use Molecule Biobox class to calculate RMSD
                self.mol.coordinates = deepcopy(crd_ref)
                self.mol.set_current(0)
                self.mol.add_xyz(crd_mdl[0])
                rmsd = self.mol.rmsd(0, 1)
            else:
                rmsd = np.sqrt(np.sum((crd_ref.flatten()-crd_mdl.flatten())**2)/crd_mdl.shape[1]) # Cartesian L2 norm

            err.append(rmsd)

        return np.array(err)


    def get_dope(self, dataset=""):

        if dataset == "":
            dataset = self.training_set    

        z = self.network.encode(dataset.float())
        decoded = self.network.decode(z)[:,:,:dataset.shape[2]]

        dope_dataset = []
        dope_decoded = []
        for i in range(dataset.shape[0]):

            # calculate DOPE score of input dataset
            crd_ref = dataset[i].permute(1,0).unsqueeze(0).data.cpu().numpy()*self.stdval + self.meanval
            self.mol.coordinates = deepcopy(crd_ref)
            self.mol.write_pdb("tmp.pdb")
            s = self._dope_score("tmp.pdb")
            dope_dataset.append(s)

            # calculate DOPE score of decoded counterpart
            crd_mdl = decoded[i].permute(1,0).unsqueeze(0).data.cpu().numpy()[:, :dataset.shape[2]]*self.stdval + self.meanval  
            self.mol.coordinates = deepcopy(crd_mdl)
            self.mol.write_pdb("tmp.pdb")
            s = self._dope_score("tmp.pdb")
            dope_decoded.append(s)

        return dope_dataset, dope_decoded


    def _get_sampling_ranges(self, samples):
        
        bx = (np.max(self.z_training[:, 0]) - np.min(self.z_training[:, 0]))*0.1 # 10% margins on x-axis
        by = (np.max(self.z_training[:, 1]) - np.min(self.z_training[:, 1]))*0.1 # 10% margins on y-axis
        xvals = np.linspace(np.min(self.z_training[:, 0])-bx, np.max(self.z_training[:, 0])+bx, samples)
        yvals = np.linspace(np.min(self.z_training[:, 1])-by, np.max(self.z_training[:, 1])+by, samples)
    
        return xvals, yvals
        
    
    def scan_error_from_target(self, target, samples=50):
        '''
        experimental function, creating a coloured landscape of RMSD vs single target structure
        target should be a Tensor of a single protein stucture loaded via load_test
        '''
        target = target.numpy().flatten()*self.stdval + self.meanval
        
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        surf_compare = np.zeros((len(self.xvals), len(self.yvals)))

        with torch.no_grad():

            for x, i in enumerate(self.xvals):
                for y, j in enumerate(self.yvals):

                    # take latent space coordinate (1) and decode it (2)
                    z = torch.tensor([[[i,j]]]).float()
                    s = self.network.decode(z)[:,:,:self.training_set.shape[2]]*self.stdval + self.meanval

                    surf_compare[x,y] = np.sum((s.numpy().flatten()-target)**2)

        #for "true Cartesian RMSD error", should multiply by [sum stdev**2]
        return np.sqrt(surf_compare/len(target)) # Cartesian L2 norm, self.xvals, self.yvals
        
    
    def scan_error(self, samples = 50):
        '''
        grid sample the latent space on a samples x samples grid (50 x 50 by default).
        Boundaries are defined by training set projections extrema, plus/minus 10%
        '''
        
        if hasattr(self, "surf_z"):
            if samples == len(self.surf_z):
                return self.surf_z, self.surf_c, self.xvals, self.yvals
        
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        surf_z = np.zeros((len(self.xvals), len(self.yvals))) # L2 norms in latent space ("drift")
        surf_c = np.zeros((len(self.xvals), len(self.yvals))) # L2 norms in Cartesian space

        with torch.no_grad():

            for x, i in enumerate(self.xvals):
                for y, j in enumerate(self.yvals):

                    # take latent space coordinate (1) and decode it (2)
                    z1 = torch.tensor([[[i,j]]]).float()
                    s1 = self.network.decode(z1)[:,:,:self.training_set.shape[2]]

                    # take the decoded structure, re-encode it (3) and then decode it (4)
                    z2 = self.network.encode(s1)
                    s2 = self.network.decode(z2)[:,:,:self.training_set.shape[2]]

                    surf_z[x,y] = np.sum((z2.numpy().flatten()-z1.numpy().flatten())**2) # Latent space L2, i.e. (1) vs (3)
                    surf_c[x,y] = np.sum((s2.numpy().flatten()-s1.numpy().flatten())**2) # Cartesian L2, i.e. (2) vs (4)

                    
        self.surf_c = np.sqrt(surf_c)
        self.surf_z = np.sqrt(surf_z)
        
        return self.surf_z, self.surf_c, self.xvals, self.yvals


    def _dope_score(self, fname):
        env = Environ()
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')
        mdl = complete_pdb(env, fname)
        atmsel = Selection(mdl.chains[0])
        score = atmsel.assess_dope()
        return score


    def scan_dope(self, samples = 50):

        if hasattr(self, "surf_dope"):
            if samples == len(self.surf_dope):
                return self.surf_dope, self.xvals, self.yvals
        
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        
        surf_dope = np.zeros((len(self.xvals), len(self.yvals)))
        with torch.no_grad():

            for x, i in enumerate(self.xvals):
                for y, j in enumerate(self.yvals):

                    # take latent space coordinate (1) and decode it (2)
                    z1 = torch.tensor([[[i,j]]]).float()
                    s1 = self.network.decode(z1)[:,:,:self.training_set.shape[2]]

                    crd_mdl = s1[0].permute(1,0).unsqueeze(0).data.cpu().numpy()[:, :self.training_set.shape[2]]*self.stdval + self.meanval  
                    self.mol.coordinates = deepcopy(crd_mdl)
                    self.mol.write_pdb("tmp.pdb")
                    surf_dope[x,y] = self._dope_score("tmp.pdb")

        self.surf_dope = surf_dope
        return surf_dope, self.xvals, self.yvals

    
    def generate(self, crd):
        '''
        generate a collection of protein conformations, given (Nx2) coordinates in the latent space
        ''' 
        with torch.no_grad():
            z = torch.tensor(crd.transpose(1, 2, 0)).float()   
            s = self.network.decode(z)[:, :, :self.training_set.shape[2]].numpy().transpose(0, 2, 1)

        return s*self.stdval + self.meanval



class MolearnGUI(object):
    
    def __init__(self, MA):
        
        if not isinstance(MA, MolearnAnalysis):
            raise Exception(f'Expecting an MolearnAnalysis instance, {type(MA)} found')
        
        self.MA = MA
        self.waypoints = [] # collection of all saved waypoints
        self.run()

        
    def oversample(self, crd, pts=10):
        '''
        add extra equally spaced points between a list of points ("pts" per interval)
        ''' 
        pts += 1
        steps = np.linspace(1./pts, 1, pts)
        pts = [crd[0,0]]
        for i in range(1, len(crd[0])):
            for j in steps:
                newpt = crd[0, i-1] + (crd[0, i]-crd[0, i-1])*j
                pts.append(newpt)

        return np.array([pts])

        
    def on_click(self, trace, points, selector):
        '''
        control display of training set
        ''' 

        # add new waypoint to list
        pt = np.array([[points.xs[0], points.ys[0]]])
        if len(self.waypoints) == 0:
            self.waypoints = pt    
        else:
            self.waypoints = np.concatenate((self.waypoints, pt))     

        # update latent space plot
        self.latent.data[3].x = self.waypoints[:, 0]
        self.latent.data[3].y = self.waypoints[:, 1]
        self.latent.update()

        # update textbox (triggering update of 3D representation)
        try:
            pt = np.array([self.latent.data[3].x, self.latent.data[3].y]).T.flatten().round(decimals=4).astype(str)
            self.mybox.value = " ".join(pt)
        except:
            return

    def interact_3D(self, mybox, samplebox):
        '''
        generate and display proteins according to latent space trail
        ''' 

        # get latent space path
        try:
            crd = np.array(mybox.split()).astype(float)
            crd = crd.reshape((1, int(len(crd)/2), 2))       
            crd = self.oversample(crd, pts=int(samplebox))
        except Exception as e:
            self.button_pdb.disabled = False
            return 

        # generate structures along path
        t = time.time()
        gen = self.MA.generate(crd)
        print(f'{crd.shape[1]} struct. in {time.time()-t:.4f} sec.')

        # display generated structures
        self.MA.mymol.load_new(gen)
        view = nv.show_mdanalysis(self.MA.mymol)
        view.add_representation("spacefill")
        display(view)

        self.button_pdb.disabled = False


    def drop_background_event(self, change):
        '''
        control colouring style of latent space surface
        '''

        state_choice = change.new

        if change.new == "drift":
            try:
                data = np.log(self.MA.surf_z).T
            except:
                return
            
            self.block0.children[2].readout_format = '.1f'

        elif change.new == "RMSD":
            try:
                data = np.log(self.MA.surf_c).T
            except:
                return
            
            self.block0.children[2].readout_format = '.1f'

        elif change.new == "DOPE":
            try:
                data = self.MA.surf_dope.T
            except:
                return
            self.block0.children[2].readout_format = 'd'
            
                
        self.latent.data[0].z = data
        self.latent.data[0].zmin = np.min(data)
        self.latent.data[0].zmax = np.max(data)
        self.block0.children[2].min = np.min(data)
        self.block0.children[2].max = np.max(data)
        self.block0.children[2].value = (np.min(data), np.max(data))
            
        self.latent.update()


    def check_training_event(self, change):
        '''
        control display of training set
        ''' 
        state_choice = change.new
        self.latent.data[1].visible = state_choice
        self.latent.update()


    def check_test_event(self, change):
        '''
        control display of test set
        ''' 
        state_choice = change.new
        self.latent.data[2].visible = state_choice
        self.latent.update()


    def range_slider_event(self, change):
        '''
        update surface colouring upon manipulation of range slider
        '''
        self.latent.data[0].zmin = change.new[0]
        self.latent.data[0].zmax = change.new[1]
        self.latent.update()


    def mybox_event(self, change):
        '''
        control manual update of waypoints
        '''

        try:
            crd = np.array(change.new.split()).astype(float)
            crd = crd.reshape((int(len(crd)/2), 2))
        except:
            self.button_pdb.disabled = False
            return

        self.waypoints = crd.copy()

        self.latent.data[3].x = self.waypoints[:, 0]
        self.latent.data[3].y = self.waypoints[:, 1]
        self.latent.update()


    def button_pdb_event(self, check):
        '''
        save PDB file corresponding to the interpolation shown in the 3D view
        '''

        root = Tk()
        root.withdraw()                                        # Hide the main window.
        root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
        fname = filedialog.asksaveasfilename(defaultextension="pdb", filetypes=[("PDB file", "pdb")])

        if fname == "":
            return

        crd = np.array(mybox.value.split()).astype(float)
        crd = crd.reshape((1, int(len(crd)/2), 2))       
        crd = oversample(crd, pts=int(samplebox.value))

        gen = generate(network, crd, stdval, meanval)
        self.MA.mymol.load_new(gen)
        protein = self.MA.mymol.select_atoms("all")

        with mda.Writer(fname, protein.n_atoms) as W:
            for ts in self.MA.mymol.trajectory:
                W.write(protein)    


    #####################################################

    def run(self):
        
        ### MENU ITEMS ###
        
        # surface representation menu
        options = []
        if hasattr(self.MA, "surf_z"):
            options.append("drift")
        if hasattr(self.MA, "surf_c"):
            options.append("RMSD")       
        if hasattr(self.MA, "surf_dope"):
            options.append("DOPE")
        if len(options) == 0:
            options.append("none")
        
        self.drop_background = widgets.Dropdown(
            options=options,
            value=options[0],
            description='Surf.:',
            layout=Layout(flex='1 1 0%', width='auto'))

        if "none" in options:
            self.drop_background.disabled = True
        
        self.drop_background.observe(self.drop_background_event, names='value')

        # training set visualisation menu
        self.check_training = widgets.Checkbox(
            value=False,
            description='show training',
            disabled=False,
            indent=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.check_training.observe(self.check_training_event, names='value')

        # test set visualisation menu
        self.check_test = widgets.Checkbox(
            value=False,
            description='show test',
            disabled=False,
            indent=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.check_test.observe(self.check_test_event, names='value')

        # text box holding current coordinates
        self.mybox = widgets.Textarea(placeholder='coordinates',
                                 description='crds:',
                                 disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.mybox.observe(self.mybox_event, names='value')

        self.samplebox = widgets.Text(value='10',
                                 description='sampling:',
                                 disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        # button to save PDB file
        self.button_pdb = widgets.Button(
            description='Save PDB',
            disabled=True, layout=Layout(flex='1 1 0%', width='auto'))

        self.button_pdb.on_click(self.button_pdb_event)

        
        # latent space range slider
        self.range_slider = widgets.FloatRangeSlider(
            description='cmap range:',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f', layout=Layout(flex='1 1 0%', width='auto'))

        self.range_slider.observe(self.range_slider_event, names='value')

        
        ### LATENT SPACE REPRESENTATION ###

        # coloured background
        if "drift" in options:
            sc = np.log(self.MA.surf_z)
        elif "DOPE" in options:
            sc = np.log(self.MA.surf_dope)
        else:
            sc = []
            
        if len(sc)>0:
            plot1 = go.Heatmap(x=self.MA.xvals, y=self.MA.yvals, z=sc.T, zmin=np.min(sc), zmax=np.max(sc),
                               colorscale='viridis', name="latent_space")   
        else:
            xvals, yvals = self.MA._get_sampling_ranges(50)
            surf_empty = np.zeros((len(xvals), len(yvals)))
            plot1 = go.Heatmap(x=xvals, y=yvals, z=surf_empty, opacity=0.0, showscale=False, name="latent_space")   
                      
        # training set
        if hasattr(self.MA, "z_training"):
            color = "white" if len(sc)>0 else "black"
            plot2 = go.Scatter(x=self.MA.z_training[:, 0], y=self.MA.z_training[:, 1],
                   showlegend=False, opacity=0.9, mode="markers",
                   marker=dict(color=color, size=5), name="training", visible=False)
        else:
            plot2 = go.Scatter(x=[], y=[])
            self.check_training.disabled = True
            
        # test set
        if hasattr(self.MA, "z_test"):
            plot3 = go.Scatter(x=self.MA.z_test[:, 0], y=self.MA.z_test[:, 1],
                   showlegend=False, opacity=0.9, mode="markers",
                   marker=dict(color='silver', size=5), name="test", visible=False)
        else:
            plot3 = go.Scatter(x=[], y=[])
            self.check_test.disabled = True
      
        # path
        plot4 = go.Scatter(x=np.array([]), y=np.array([]),
                   showlegend=False, opacity=0.9,
                   marker=dict(color='red', size=7))

        self.latent = go.FigureWidget([plot1, plot2, plot3, plot4])
        self.latent.update_layout(xaxis_title="latent vector 1", yaxis_title="latent vector 2",
                         autosize=True, width=400, height=350, margin=dict(l=75, r=0, t=25, b=0))
        self.latent.update_xaxes(showspikes=False)
        self.latent.update_yaxes(showspikes=False)       

        if len(sc)>0:
            self.range_slider.value = (np.min(sc), np.max(sc))
            self.range_slider.min = np.min(sc)
            self.range_slider.max = np.max(sc)
            self.range_slider.step = (np.max(sc)-np.min(sc))/100.0
            self.range_slider.disabled = False

        # 3D protein representation (triggered by update of textbox)
        self.protein = widgets.interactive_output(self.interact_3D, {'mybox': self.mybox, 'samplebox': self.samplebox})

        
        ### WIDGETS ARRANGEMENT ###
        
        self.block0 = widgets.VBox([self.check_training, self.check_test, self.range_slider, self.drop_background, self.samplebox, self.mybox, self.button_pdb],
                              layout=Layout(flex='1 1 2', width='auto', border="solid"))

        self.block1 = widgets.VBox([self.latent], layout=Layout(flex='1 1 auto', width='auto'))
        self.latent.data[0].on_click(self.on_click)

        self.block2 = widgets.VBox([self.protein], layout=Layout(flex='1 5 auto', width='auto'))

        self.scene = widgets.HBox([self.block0, self.block1, self.block2])
        self.scene.layout.align_items = 'center'

        display(self.scene)
