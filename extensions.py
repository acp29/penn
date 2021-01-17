"""
User-defined Python extensions that can be called from the menu.
"""

import spells
import penn

# class to create a submenu of Extensions
class Extension(object):
    """
    A Python extension that can be added as a submenu in 
    the Extensions menu of Stimfit.
    """
    def __init__(self, menuEntryString, pyFunc, description="", 
                 requiresFile=True, parentEntry=None):
        """
        Arguments:
        menuEntryString -- This will be shown as a menu entry.
        pyFunc -- The Python function that is to be called.
                  Takes no arguments and returns a boolean.
        description -- A more verbose description of the function.
        requiresFile -- Whether pyFunc requires a file to be opened.
        """
        self.menuEntryString = menuEntryString
        self.pyFunc = pyFunc
        self.description = description
        self.requiresFile = requiresFile
        self.parentEntry = parentEntry

# define an Extension: it will appear as a submenu in the Extensions Menu
myExt00 = Extension("Load ACQ4", penn.analysis.loadacq4, "Load ACQ5 hdf5 files (.ma)", False)
myExt01 = Extension("Load PHY", penn.analysis.loadphy, "Load ephysIO formatted HDF5 (Matlab v7.3) files (.mat)", False)
myExt02 = Extension("Save PHT", penn.analysis.savephy, "Save ephysIO formatted HDF5 (Matlab v7.3) files (.mat)", True)
myExt03 = Extension("Count APs", spells.count_aps, "Count events >0 mV in selected traces", True)
myExt04 = Extension("crop", penn.analysis.crop, "Crop all traces to the fit cursor positions", True)
myExt05 = Extension("blankstim", penn.analysis.blankstim, "Blank values between fit cursors in all traces", True)
myExt06 = Extension("interpstim", penn.analysis.interpstim, "Interpolate values between fit cursors in all traces", True)
myExt07 = Extension("peakscale", penn.analysis.peakscale, "Scale the selected traces to their mean peak amplitude.", True)
myExt08 = Extension("monoexpfit", penn.analysis.monoexpfit, "Fit an exponential with offset to current trace between fit cursors", True)
myExt09 = Extension("biexpfit", penn.analysis.biexpfit, "Fit a sum of 2 exponentials with offset to current trace between fit cursors", True)
myExt10 = Extension("peakalign", penn.analysis.peakalign, "Align selected traces in the active window to the peak index", True)
myExt11 = Extension("risealign", penn.analysis.risealign, "Align selected traces in the active window to the rtlow index", True)
myExt12 = Extension("rmean3traces", penn.analysis.rmean3traces, "Calculate ensemble mean trace of every 3 traces in the active channel", True)
myExt13 = Extension("trainpeaks", penn.analysis.trainpeaks, "Measure peaks in train in the current trace in the active channel", True)
myExt14 = Extension("reverse", penn.analysis.reverse, "Reverse the order of all traces", True)
myExt15 = Extension("mean_every_9th", penn.analysis.mean_every_9th, "Mean of the first and every 9th trace", True)
myExt16 = Extension("upsample_flex", penn.analysis.upsample_flex, "Upsample FlexStation traces to 1 ms intervals by interpolation", True)
myExt17 = Extension("batch_integration", penn.analysis.batch_integration, "Perform batch trapezium integration between fit cursors in active window ", True)
myExt18 = Extension("whole-cell properties", penn.analysis.wcp, "Measure whole cell properties from voltage-step current transient (-5 mV, 10 ms start, 20 ms long)", False)
myExt19 = Extension("subtract_base", penn.analysis.subtract_base, "Measure whole cell properties from voltage-step current transient (-5 mV, 10 ms start, 20 ms long)", False)

extensionList = [myExt00,myExt01,myExt02,myExt03,myExt04,myExt05,myExt06,myExt07,myExt08,myExt09,myExt10,myExt11,myExt12,myExt13,myExt14,myExt15,myExt16,myExt17,myExt18,myExt19,]
