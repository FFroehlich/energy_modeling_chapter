#!/usr/bin/env python
# coding: utf-8

# # A computational framework for the efficient formulation and calibration of large-scale, thermodynamically consistent kinetic models and its application

# ## Abstract

# 

# ## Introduction

# Cells use signal transduction to respond to extracellular cues. Extracellular ligands bind to RTKs that initiate phosphorylation cascades. Depending on protein abundance, different protein complexes will be assembled, which determines which part of downstream signaling will be activated. In contrast to phosphorylation, assembly of protein complexes does not consume energy. Consumption of energy can be checked via wegscheider cycle conditions. Enumerating all cycles is an NP-hard problem, so structured approach is necessary to ensure energy conservation. Faeder et al have introduced thermodynamic rule based modelling which derives kinetic rates from the specified gibbs free energies. Binding reactions generated using thermodynamic framework inherently satisfy energy conservation, making them biophysically more plausible. Also allows specification of rules that consume energy, such as phosphorylation or dephosphorylation by explicitely specifying consumption of cellular energy sources such as ATP or NADPH.
# Derivation of kinetic rates from gibbs free energies are carried out using arrhenius equation, but more modern formulations exist. Formulation provides an intuition why log-scale is more natural scale for most parameters and why optimization etc works better when log-transformed.

# Energy conservation and corresponding wegscheider conditions introduce an constraint on parameters, which may suggest that this yields smaller less complex models. However, thermodynamically sounds models require reversibility of all reactions, which usually leads to more complex models. Specifically, binarization of reactions usually leads to non-reversible reactions that may violate energy conservation. An example for binarization of a reaction would be binding of RAS. RAF has a much higher affinity to gtp loaded Ras (RAS-gtp) compared to gdp loaded Ras (RAS-gdp), so it is tempting to implement a conditional reaction where RAF only binds RAS-gtp, but not RAS-gdp. As RAS is a GTPase, it can hydrolyse the loaded gtp to gdp, converting RAS-gtp to RAS-gdp. This reaction creates a complex in which RAS-gdp is bound to RAF. As the corresponding binding reaction was explicitely not created, the unbinding reaction for RAS-gdp and RAF would necessarily be irreversible and may thus violate energy conservation. Implementing a non-binarized reaction where RAF binds both RAS-gtp and RAS-gdp can easily be done in the thermodynamic framework, where the difference in affinity would be implemented as difference in Gibbs Free Energy, and would be guaranteed to satisfy wegscheider cycle conditions and thus satisfy energy conservation. However, the non-binarized, thermodynamic model needs to account for more species, which may pose computational challenges when the model describes more complex oligomerization processes. 

# Tools such as NFSim are particularly suited to simulate 

# ## Formulating a Thermodynamic Model of RAF inhibition in PySB

# First import various classes that are necessary to define rules and initialize the model

# In[1]:


from pysb import Model, Compartment, Rule, EnergyPattern, Expression
model = Model('thermo_raf');


# ### Protein Species
# As first step of the model construction, we have to define all molecules which we want to account for. In PySB, this is done through using the Monomer class. Each monomer is initialized by a unique identifier, a list of site as well as, optionally, a list of states for each site. Sites without states are typically describe the bonds involved in complex formation, while sites with states can also describe (post-translational) protein modifitations such as mutations or phosphorylation. 
# 
# Here, we we will define monomers _BRAF_ and _CRAF_ which both feature the interaction sites _RBD_ (interaction domain with RAS), _mek_ (interaction domain with MEK), _raf_ (dimerization domain) and _rafi_ (inhibitor binding site). For BRAF we also add a mutation sites for the 600th amino acid, which can take values _V_ (valine, wildtype) or _E_ (glutamic acid, oncogenic mutation).

# In[2]:


from pysb import Monomer
Monomer('BRAF', ['AA600', 'RBD', 'mek', 'raf', 'rafi'], {'AA600': ['E', 'V']})
Monomer('CRAF', ['RBD', 'mek', 'raf', 'rafi']);


# Next, we will specify initial concentrations. For this model, the concentration of all molecular species will be in $\mu\mathrm{M}$. However, we will use protein counts per cell derived from absolute proteomics to inform initial conditions. Accordingly, parameters that define initial concentrations have to be transformed from molecule per cell to $\mu\mathrm{M}$, which is achieved by dividing them by cell volume (here assumed to be $1pL = 10^{-12} L$) and Avogadro constant ($6.022 10^{23}$ molecules) and multiplying with $10^{6}$ to account for the unit prefix $\mu$.
# 
# In the following code we first introduce two expressions that define the avogadro constant and cell volume. PySB automatically creates workspace variables with the name of the expressions, which is passed as first argument, that can then later be used as argument to functions or as part of more complex expressions. Next, we introduce two parameter _BRAF_0_ and _CRAF_0_ that define initial abundances as molecules per cell and convert these abundances to concentrations in $\mu\mathrm{M}$ using the expressions _initBRAF_ and _initCRAF_. These expressions are then used to define the initial abundances of two molecular species for CRAF and BRAF. A molecular species is defined by a pattern, which is created by invoking the respective monomer with the state of each site as keyword argument. For initial conditions, the respective patterns have to be explicit, i.e., all states have to defined. Here, we specify that all sites are unbound (denoted by _None_), with the exception of the the _AA600_ site, for which we have to pick ony of the previously specified states, where we pick the oncogenic variant denoted by _E_.

# In[3]:


from pysb import Parameter, Expression, Initial
import sympy as sp

# define Avogadro constant and volume as hardcoded expressions
Expression('N_Avogadro', sp.Float(6.02214076e+23))
Expression('volume', sp.Float(1e-12))

# define initial abundance parameters 
Parameter('BRAF_0', 0.0)
Parameter('CRAF_0', 0.0)

# convert initial abundances to concentrations
Expression('initBRAF', 1000000.0*BRAF_0/(N_Avogadro*volume))
Expression('initCRAF', 1000000.0*CRAF_0/(N_Avogadro*volume))

# define initial molecular species
Initial(BRAF(AA600='E', RBD=None, mek=None, raf=None, rafi=None), initBRAF)
Initial(CRAF(RBD=None, mek=None, raf=None, rafi=None), initCRAF);


# ### Protein Interactions
# In rule-based modeling, all dynamic interactions are specified as rules. Rules are generalizations of biochemical reactions and their action is defined by a reactant pattern and a product pattern. When applied to a molecular species, the action of the rule, i.e., the respective biochemical reaction, is implemented by replacing one occurence of the reactant pattern by the product pattern in a molecular species. The reactant pattern and product pattern do not have to be explicit and a single rule can define multiple different biochemical reactions, depending in how often and in how many molecular species the reactant pattern occurs.
# 
# Non-thermodynamic rules additionally require forward and, if applicable, reverse reaction rates. In contrast, thermodynamic rules are specified in terms of activation energy and the phenomenological constant $\phi$, which encodes whether changes in free energy affect forwards or reverse rates. For thermodynamic rules, the reaction rate also depends on EnergyPatterns that may apply to educts or products of the reaction. This means that all reactions generated by non-thermodynamic rules have the same reaction rates, unless the rate is a local function. In contrast, reactions generated by thermodynamic rules can have context dependent reaction rates.
# 
# In the following we will specify the dimerization of RAF molecules. We introduce three parameters, the forward rate $k_f$, the dissociation constant $k_D$ and the thermodynamic parameter $\phi$. 

# In[4]:


Parameter('bind_RAF_RAF_Ea', 10.0)
Parameter('bind_RAF_RAF_dG', 0.01)
Parameter('bind_RAF_RAF_phi', 1.0)

Expression('Ea0_bind_RAF_RAF', -bind_RAF_RAF_phi*bind_RAF_RAF_dG - bind_RAF_RAF_Ea);


# To construct homo- and heterodimerization rules for all RAF paralogs, we use itertools to loop over all combinations of BRAF and CRAF. This assumes equal affinities and dynamics for all proteins.

# In[5]:


import itertools as itt

for RAF1, RAF2 in itt.combinations_with_replacement([BRAF, CRAF], 2):
    Rule(f'{RAF1.name}_and_{RAF2.name}_bind_and_dissociate', 
         RAF1(raf=None) + RAF2(raf=None) | RAF1(raf=1) % RAF2(raf=1), 
         bind_RAF_RAF_phi, Ea0_bind_RAF_RAF, energy=True)
    EnergyPattern(f'ep_bind_{RAF1.name}_{RAF2.name}', 
                  RAF1(raf=1) % RAF2(raf=1), bind_RAF_RAF_dG);


# ### RAF inhibitor
# Here we introduce a RAF inhibitor RAFi. We assume that the inhibitor is added to the cell medium at some point and quickly diffuses in and out of the cell. As extracellular space is much bigger than the volume of a cell, we can assume an infinite reservoir of molecules on the outside and assume that the intracellular inhibitors concentration will be unaffected by intracellular reactions. Accordingly, we specify the respective initial as constant, which means that the respective molecular species can participate in reactions, but it's concentration will remain constant.

# In[6]:


Monomer('RAFi', ['raf'])

Initial(RAFi(raf=None), Parameter('RAFi_0', 0.0), fixed=True);


# Next we define binding reactions for RAFi with both BRAF and CRAF, again assuming the same affinities and activation energies for both reactions.

# In[7]:


Parameter('bind_RAFi_RAF_Ea', 10.0)
Parameter('bind_RAFi_RAF_dG', 0.01)
Parameter('bind_RAFi_RAF_phi', 1.0)
Expression('Ea0_bind_RAFi_RAF', -bind_RAFi_RAF_phi*bind_RAFi_RAF_dG - bind_RAFi_RAF_Ea)

for RAF in [BRAF, CRAF]:
    Rule(f'RAFi_and_{RAF.name}_bind_and_dissociate', 
         RAFi(raf=None) + RAF(rafi=None) | RAFi(raf=1) % RAF(rafi=1), 
         bind_RAFi_RAF_phi, Ea0_bind_RAFi_RAF, energy=True)
    EnergyPattern(f'ep_bind_{RAF.name}_RAFi', RAFi(raf=1) % RAF(rafi=1), bind_RAFi_RAF_dG);


# ### Paradoxical Activation
# RAF inhibitors inhibit growth for BRAF mutants but promote growth for NRAS mutants. At the strucural level, this can be rationalized by assuming that RAF inhibitors have higher affinity towards drug-unbound RAF dimers and lower affinity towards drug bound RAF dimers. The symmetry conveyed by the no energy consuming nature of molecular binding reactions, implies that RAF inhibitors, at low to medium concentrations, promote dimerization but incompletely inhibit signaling as they only bind to one of the two protomers in a dimer. As MAPK signaling in NRAS mutant cells is mediated by RAF dimers, respective signaling is amplified, leading to increased growth. In the thermodynamic framework, this can be implemented by adding an energy pattern that controls the Gibbs free energy of RAF-RAF-RAFi trimers. Note that we do not specify how these trimers are formed, so the change in energy will equally apply to the rates of all reactions that produce these trimers. In this example, an decrease in energy would equally increase RAF dimerization and inhbitor binding, thus implementing the previously described symmetry.

# In[8]:


Parameter('ep_RAF_RAF_mod_RAFi_single_ddG', 0.001)
for RAF1, RAF2 in itt.product([BRAF, CRAF], repeat=2):
    EnergyPattern(f'ep_{RAF1.name}_{RAF2.name}_mod_RAFi_single', 
                  RAF1(raf=1, rafi=None) % RAF2(raf=1, rafi=2) % RAFi(raf=2), ep_RAF_RAF_mod_RAFi_single_ddG);


# To implement the protection of the second dimer, we add additional energy patterns that change the Gibbs free energy of RAFi-RAF-RAF-RAFi quatramers.

# In[9]:


Parameter('ep_RAF_RAF_mod_RAFi_double_ddG', 1000.0)
for RAF1, RAF2 in itt.combinations_with_replacement([BRAF, CRAF], r=2):
    EnergyPattern(f'ep_{RAF1.name}_{RAF2.name}_mod_RAFi_double', 
                  RAFi(raf=2) % BRAF(raf=1, rafi=2) % BRAF(raf=1, rafi=3) % RAFi(raf=3), 
                  ep_RAF_RAF_mod_RAFi_double_ddG);


# We will now load the remainder of the model from a file. Briefly, the full model describes MEK and ERK phosphorylation downstream as well as EGF stimulatable EGFR signaling upstream of RAF signaling. Moreover, it incorporates negative feedback from ERK on both MAPK and EGFR signaling. A comprehensive description of this part of the model is available in REF. Note however, that parts of the model were substantially simplified account for the reduced set of experimental data.

# In[10]:


from pysb import Observable, ANY
def extend_model():
    Monomer('RAS', ['raf', 'state'], {'state': ['gdp', 'gtp']})
    Monomer('MEK', ['Dsite', 'phospho', 'raf'], {'phospho': ['p', 'u']})
    Monomer('ERK', ['CD', 'phospho'], {'phospho': ['p', 'u']})
    Monomer('DUSP', ['erk'])

    Parameter('ep_RAF_RAF_mod_RASstategtp_double_ddG', 1000.0)
    Parameter('bind_RASstategtp_RAF_Ea', 10.0)
    Parameter('bind_RASstategtp_RAF_dG', 0.01)
    Parameter('bind_RASstategtp_RAF_phi', 1.0)
    Parameter('bind_RAF_MEKphosphou_Ea', 10.0)
    Parameter('bind_RAF_MEKphosphou_dG', 0.01)
    Parameter('bind_RAF_MEKphosphou_phi', 1.0)
    Parameter('catalyze_RAF_RAFrafiNone_MEK_p_kcat', 10.0)
    Parameter('catalyze_RAFrafiNone_MEK_p_kcat', 10.0)
    Parameter('bind_MEK_ERKphosphou_kf', 10.0)
    Parameter('bind_MEK_ERKphosphou_kD', 0.01)
    Parameter('catalyze_MEKmekiNone_phosphop_ERK_p_kcat', 10.0)
    Parameter('bind_DUSP_ERKphosphop_kf', 10.0)
    Parameter('bind_DUSP_ERKphosphop_kD', 0.01)
    Parameter('synthesize_ERKphosphop_DUSP_ERK_gexpslope', 1000.0)
    Parameter('synthesize_ERKphosphop_DUSP_kdeg', 10.0)
    Parameter('synthesize_ERKphosphop_DUSP_ERK_kM', 0.0)
    Parameter('inhibition_ERKphosphop_RAS_kM', 0.0)
    Parameter('catalyze_PP2A_MEK_u_kcatr', 1.0)
    Parameter('catalyze_DUSP_ERK_u_kcatr', 1.0)
    Parameter('activation_RAS_kcat', 1.0)

    Parameter('RAS_0', 0.0)
    Parameter('MEK_0', 0.0)
    Parameter('ERK_0', 0.0)
    Parameter('EGF_0', 0.0)
    Parameter('DUSP_eq', 10000.0)

    Expression('Ea0_bind_RASstategtp_BRAF', 
               -bind_RASstategtp_RAF_phi*bind_RASstategtp_RAF_dG - bind_RASstategtp_RAF_Ea)
    Expression('Ea0_bind_RASstategtp_CRAF', 
               -bind_RASstategtp_RAF_phi*bind_RASstategtp_RAF_dG - bind_RASstategtp_RAF_Ea)
    Expression('Ea0_bind_BRAF_MEKphosphou', 
               -bind_RAF_MEKphosphou_phi*bind_RAF_MEKphosphou_dG - bind_RAF_MEKphosphou_Ea)
    Expression('Ea0_bind_CRAF_MEKphosphou', 
               -bind_RAF_MEKphosphou_phi*bind_RAF_MEKphosphou_dG - bind_RAF_MEKphosphou_Ea)

    Expression('bind_MEK_ERKphosphou_kr', bind_MEK_ERKphosphou_kD*bind_MEK_ERKphosphou_kf)
    Expression('bind_DUSP_ERKphosphop_kr', bind_DUSP_ERKphosphop_kD*bind_DUSP_ERKphosphop_kf)
    Expression('catalyze_PP2A_MEK_u_kcat', catalyze_PP2A_MEK_u_kcatr*catalyze_RAFrafiNone_MEK_p_kcat)
    Expression('catalyze_DUSP_ERK_u_kcat', catalyze_DUSP_ERK_u_kcatr*catalyze_MEKmekiNone_phosphop_ERK_p_kcat)

    Expression('initMEK', 1000000.0*MEK_0/(N_Avogadro*volume))
    Expression('initERK', 1000000.0*ERK_0/(N_Avogadro*volume))
    Expression('initRAS', 1000000.0*RAS_0/(N_Avogadro*volume)*EGF_0/100.0)

    Observable('modulation_ERKphosphop', ERK(phospho='p'))

    Expression('synthesize_ERKphosphop_DUSP_ERK_kmod', modulation_ERKphosphop*synthesize_ERKphosphop_DUSP_ERK_gexpslope/(modulation_ERKphosphop + synthesize_ERKphosphop_DUSP_ERK_kM) + 1)
    Expression('synthesize_ERKphosphop_DUSP_ksyn', 1.0*DUSP_eq*synthesize_ERKphosphop_DUSP_kdeg*synthesize_ERKphosphop_DUSP_ERK_kmod)
    Expression('inhibition_ERKphosphop_RAS_kmod', activation_RAS_kcat/(modulation_ERKphosphop + inhibition_ERKphosphop_RAS_kM))


    Rule('RASgtp_and_BRAF_bind_and_dissociate', RAS(raf=None, state='gtp') + BRAF(RBD=None) | RAS(raf=1, state='gtp') % BRAF(RBD=1), bind_RASstategtp_RAF_phi, Ea0_bind_RASstategtp_BRAF, energy=True)
    Rule('RASgtp_and_CRAF_bind_and_dissociate', RAS(raf=None, state='gtp') + CRAF(RBD=None) | RAS(raf=1, state='gtp') % CRAF(RBD=1), bind_RASstategtp_RAF_phi, Ea0_bind_RASstategtp_CRAF, energy=True)
    Rule('BRAF_and_uMEK_bind_and_dissociate', BRAF(mek=None) + MEK(phospho='u', raf=None) | BRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_MEKphosphou_phi, Ea0_bind_BRAF_MEKphosphou, energy=True)
    Rule('CRAF_and_uMEK_bind_and_dissociate', CRAF(mek=None) + MEK(phospho='u', raf=None) | CRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_MEKphosphou_phi, Ea0_bind_CRAF_MEKphosphou, energy=True)
    Rule('BRAF_BRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % BRAF(RBD=ANY, mek=1, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + BRAF(RBD=ANY, mek=None, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
    Rule('BRAF_CRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % BRAF(RBD=ANY, mek=1, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + BRAF(RBD=ANY, mek=None, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
    Rule('CRAF_BRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % CRAF(RBD=ANY, mek=1, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + CRAF(RBD=ANY, mek=None, raf=2, rafi=None) % BRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
    Rule('CRAF_CRAF_phosphorylates_MEK', MEK(phospho='u', raf=1) % CRAF(RBD=ANY, mek=1, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2) >> MEK(phospho='p', raf=None) + CRAF(RBD=ANY, mek=None, raf=2, rafi=None) % CRAF(RBD=ANY, raf=2), catalyze_RAF_RAFrafiNone_MEK_p_kcat)
    Rule('BRAFV600E_phosphorylates_MEK_bound1', MEK(phospho='u', raf=1) % BRAF(AA600='E', mek=1, raf=None, rafi=None) >> MEK(phospho='p', raf=None) + BRAF(AA600='E', mek=None, raf=None, rafi=None), catalyze_RAFrafiNone_MEK_p_kcat)
    Rule('BRAFV600E_phosphorylates_MEK_bound2', MEK(phospho='u', raf=1) % BRAF(AA600='E', RBD=None, mek=1, raf=ANY, rafi=None) >> MEK(phospho='p', raf=None) + BRAF(AA600='E', RBD=None, mek=None, raf=ANY, rafi=None), catalyze_RAFrafiNone_MEK_p_kcat)
    Rule('BRAFV600E_phosphorylates_MEK_bound3', MEK(phospho='u', raf=1) % BRAF(AA600='E', RBD=ANY, mek=1, raf=2, rafi=None) % BRAF(RBD=None, raf=2) >> MEK(phospho='p', raf=None) + BRAF(AA600='E', RBD=ANY, mek=None, raf=2, rafi=None) % BRAF(RBD=None, raf=2), catalyze_RAFrafiNone_MEK_p_kcat)
    Rule('BRAFV600E_phosphorylates_MEK_bound4', MEK(phospho='u', raf=1) % BRAF(AA600='E', RBD=ANY, mek=1, raf=2, rafi=None) % CRAF(RBD=None, raf=2) >> MEK(phospho='p', raf=None) + BRAF(AA600='E', RBD=ANY, mek=None, raf=2, rafi=None) % CRAF(RBD=None, raf=2), catalyze_RAFrafiNone_MEK_p_kcat)
    Rule('MEK_is_dephosphorylated', MEK(phospho='p') >> MEK(phospho='u'), catalyze_PP2A_MEK_u_kcat)
    Rule('MEK_binds_uERK', MEK(Dsite=None) + ERK(CD=None, phospho='u') >> MEK(Dsite=1) % ERK(CD=1, phospho='u'), bind_MEK_ERKphosphou_kf)
    Rule('MEK_dissociates_from_ERK', MEK(Dsite=1) % ERK(CD=1) >> MEK(Dsite=None) + ERK(CD=None), bind_MEK_ERKphosphou_kr)
    Rule('pMEK_phosphorylates_ERK', ERK(CD=1, phospho='u') % MEK(Dsite=1, phospho='p') >> ERK(CD=None, phospho='p') + MEK(Dsite=None, phospho='p'), catalyze_MEKmekiNone_phosphop_ERK_p_kcat)
    Rule('DUSP_binds_pERK', DUSP(erk=None) + ERK(CD=None, phospho='p') >> DUSP(erk=1) % ERK(CD=1, phospho='p'), bind_DUSP_ERKphosphop_kf)
    Rule('DUSP_dissociates_from_ERK', DUSP(erk=1) % ERK(CD=1) >> DUSP(erk=None) + ERK(CD=None), bind_DUSP_ERKphosphop_kr)
    Rule('DUSP_dephosphorylates_ERK', ERK(CD=1, phospho='p') % DUSP(erk=1) >> ERK(CD=None, phospho='u') + DUSP(erk=None), catalyze_DUSP_ERK_u_kcat)
    Rule('synthesis_DUSP', None >> DUSP(erk=None), synthesize_ERKphosphop_DUSP_ksyn)
    Rule('basal_degradation_DUSP', DUSP() >> None, synthesize_ERKphosphop_DUSP_kdeg, delete_molecules=True)
    Rule('activation_RAS', RAS(state='gdp') | RAS(state='gtp'), inhibition_ERKphosphop_RAS_kmod, activation_RAS_kcat)

    EnergyPattern('ep_BRAF_BRAF_mod_RAS_double', RAS(raf=2, state='gtp') % BRAF(RBD=2, raf=1) % BRAF(RBD=3, raf=1) % RAS(raf=3, state='gtp'), ep_RAF_RAF_mod_RASstategtp_double_ddG)
    EnergyPattern('ep_BRAF_CRAF_mod_RAS_double', RAS(raf=2, state='gtp') % BRAF(RBD=2, raf=1) % CRAF(RBD=3, raf=1) % RAS(raf=3, state='gtp'), ep_RAF_RAF_mod_RASstategtp_double_ddG)
    EnergyPattern('ep_CRAF_CRAF_mod_RAS_double', RAS(raf=2, state='gtp') % CRAF(RBD=2, raf=1) % CRAF(RBD=3, raf=1) % RAS(raf=3, state='gtp'), ep_RAF_RAF_mod_RASstategtp_double_ddG)
    EnergyPattern('ep_bind_RASstategtp_BRAF', RAS(raf=1, state='gtp') % BRAF(RBD=1), bind_RASstategtp_RAF_dG)
    EnergyPattern('ep_bind_RASstategtp_CRAF', RAS(raf=1, state='gtp') % CRAF(RBD=1), bind_RASstategtp_RAF_dG)
    EnergyPattern('ep_bind_BRAF_MEKphosphou', BRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_MEKphosphou_dG)
    EnergyPattern('ep_bind_CRAF_MEKphosphou', CRAF(mek=1) % MEK(phospho='u', raf=1), bind_RAF_MEKphosphou_dG)

    Initial(RAS(raf=None, state='gdp'), initRAS)
    Initial(MEK(Dsite=None, phospho='u', raf=None), initMEK)
    Initial(ERK(CD=None, phospho='u'), initERK)

extend_model();


# ## Importing data in PEtab Format

# With the model at hand, the next step for model calbitration is defining the training data. We will implement this in PEtab, which simplifies the definition of multiple experimental conditions. The petab specification of a calibration problem consists of the tables describing model observables, experimental measurements, experimental conditions and model parameters. Additionally, tables describing the visualization of simulations and data may also be included. In the following we will guide the reader through the definition

# ## Observables
# 
# We will start the petab definition by specifying the model observables. Observables define the the model quantities that were measured experimentally, here pMEK and pERK. We add respective observables to the model using pysb Observables:

# In[11]:


Observable('pMEK_obs', MEK(phospho='p'))
Observable('pERK_obs', ERK(phospho='p'));


# These observables quantify all MEK and ERK molecules that are phosphorylated on the site `phospho`, which accounts for phosphorylation on TS on MEK and on ERK.
# 
# The model quantifies pMEK and pERK in concentrations, but measurements are noise corrupted, measured in fluorescence intensity and also include background fluorescence. As the scaling between concentrations and intensity and the amount of background signal is known, we include scaling and offset parameters in the petab observable definition. To account for noise corruption, we specify a single noise parameter as noise Formula, which corresponds to Gaussian noise distribution with the respective parameter as standard deviation. The prefix `noiseParameter1` indicates that the value of the parameter will be provided in the respective column of the measurements table.

# In[12]:


import petab
import pandas as pd
observable_table = pd.DataFrame([
    {
        petab.OBSERVABLE_ID: obs_id,
        petab.OBSERVABLE_FORMULA: f'{obs_id}_scale*{obs_id}_obs + {obs_id}_offset',
        petab.NOISE_FORMULA: f'noiseParameter1_{obs_id}'
    } for obs_id in ['pMEK', 'pERK']
]).set_index(petab.OBSERVABLE_ID)
observable_table


# ## Measurements
# As experimental measurement, we will load one of the datasets from REF, which contains the response of A375 cells cultured for 24h at different concentrations of multiple RAF and MEK inhibitors. The response is measured as normalized phospho-MEK (pMEK) and phospho-ERK (pERK) abundances in both EGF stimulated and unstimulated cells after 5 minutes.

# In[13]:


import synapseclient as sc
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 6)

syn = sc.Synapse()
syn.login(silent=True);
data_df = pd.read_csv(syn.get('syn22804081').path)
data_df


# To keep compute time to a minimum, we will only consider a subset of the available data, i.e., only the experiments involving vemurafenib and dabrafenib as RAF inhibitors. The model itself only contains generic `RAFi` parameters and species, which we can now map to specific inhibitors using experimental conditions. This permits the simultaneous estimation of inhibitor specific kinetic rates in conjunction with all other model parameters. Such an multi-experiment setup can improve parameter identifiability. In the following, we will simulatenously generate the condition and measurement table. As the data is encoded in a matrix format, while PEtab requires a long format, this requires some additional processing. Briefly, the code below extracts all data points and sets what model observable they belong to, the value of the measurement, the time of measurement, the noise parameter, information about the experimental condition as well as a dataset ID, which is only used for plotting.
# 
# PEtab allows the specfication of a simulation condition as well as a preequilibration condition.

# In[14]:


RAFis = ['Vemurafenib', 'Dabrafenib', 'PLX8394', 'LY3009120', 'AZ_628']
MEKis = ['Cobimetinib', 'Trametinib', 'Selumetinib', 'Binimetinib', 'PD0325901']


conditions = dict()


def format_petab(row):
    suffixes = [''] + [f'.{idx}' for idx in range(1,10)]
    datapoints = []
    if row[MEKis].any() or row['EGF'] > 0:
        return datapoints
    # loop over columns of the data matrix
    for suffix in suffixes:
        # extract data
        datapoint = {
            petab.OBSERVABLE_ID: 'pMEK' if row.pMEK else 'pERK',
            petab.MEASUREMENT: row[f'Mean{suffix}'],
            petab.TIME: row['Time_EGF'],
            petab.NOISE_PARAMETERS: row[f'Std{suffix}'],
        }
        # extract condition information
        if (row[RAFis] != 0).sum() != 1 or ((row['Vemurafenib'] == 0) and (row['Dabrafenib'] == 0)):
            continue

        # find first nonzero
        rafi = RAFis[(row[RAFis] != 0).argmax()]

        # extract drug concentration
        drug_conc = row[f'Concentration (uM){suffix}'] if row[rafi] == -1 else row[rafi]
        # condition id must be sanitzed, must match '^[a-zA-Z]+[\w_]*$'
        drug_str = f'{rafi}_{drug_conc}'.replace('.', '_').replace('-', '_')

        condition_str = drug_str + (
            f'__EGF_{row["EGF"]}' if row['EGF'] > 0 else ''
        )
        condition = {
            petab.CONDITION_ID: condition_str,'RAFi_0': drug_conc,
            'EGF_0': row["EGF"], 'bind_RAFi_RAF_Ea': f'bind_{rafi}_RAF_Ea',
            'bind_RAFi_RAF_dG': f'bind_{rafi}_RAF_dG',
            'ep_RAF_RAF_mod_RAFi_single_ddG': f'ep_RAF_RAF_mod_{rafi}_single_ddG',
            'ep_RAF_RAF_mod_RAFi_double_ddG': f'ep_RAF_RAF_mod_{rafi}_double_ddG'
        }
        # set baseline for datapoint
        datapoint[petab.PREEQUILIBRATION_CONDITION_ID] = drug_str
        # set id for condition and datapoint
        datapoint[petab.SIMULATION_CONDITION_ID] = condition_str
        datapoint[petab.DATASET_ID] =  ('EGF__' if row['EGF'] > 0 else 'ctrl__') + rafi
        condition[petab.CONDITION_ID] = condition_str
        
        datapoints.append(datapoint)
        if condition_str not in conditions:
            conditions[condition_str] = condition
            
    return datapoints

measurement_table = pd.DataFrame([
    d
    for ir, row in data_df.iterrows()
    for d in format_petab(row)
])
condition_table = pd.DataFrame(conditions.values()).set_index(petab.CONDITION_ID);


# In[15]:


import numpy as np
for group, frame in measurement_table.groupby(petab.OBSERVABLE_ID):
    measurement_table.loc[measurement_table[petab.OBSERVABLE_ID]==group, petab.NOISE_PARAMETERS] =         np.sqrt(frame[petab.NOISE_PARAMETERS].apply(np.square).mean())


# In[16]:


measurement_table


# In[17]:


condition_table


# In[18]:


import numpy as np
condition_pars = [
    par.name
    for par in model.parameters
    if par.name in condition_table.columns
]

free_parameters = [
    par.name for par in model.parameters
    if par.name not in condition_pars
] + [
    name
    for par in condition_pars
    for name in np.unique(condition_table[par])
    if isinstance(name, str)
] + [
    'pMEK_offset','pMEK_scale', 'pERK_offset', 'pERK_scale'
]


# In[19]:


lbs = {
    'Ea': -10,
    'dG': -10,
    'ddG': -10,
    'phi': 0,
    'offset': 1e-3,
    'scale': 1e0,
    '0': 1e2,
    'eq': 1e1,
    'kcat': 1e-3,
    'kf': 1e-2,
    'kD': 1e-4,
    'gexpslope': 1e0,
    'kdeg': 1e-3,
    'kM': 1e-3,
    'kcatr': 1e-8,
    'koff': 1e-5,
}
ubs = {
    'Ea': 10,
    'dG': 10,
    'ddG': 10,
    'phi': 1,
    'offset': 1e0,
    'scale': 1e6,
    '0': 1e6,
    'eq': 1e5,
    'kcat': 1e3,
    'kf': 1e4,
    'kD': 1e1,
    'gexpslope': 1e6,
    'kdeg': 1e2,
    'kM': 1e1,
    'kcatr': 1e0,
    'koff': 1e0,
}

initials = {
    'BRAF_0': 1e3,
    'CRAF_0': 1e4,
    'MEK_0': 1e5,
    'ERK_0': 1e5,
    'RAS_0': 5e4,
}


parameter_table = pd.DataFrame([
    {
        petab.PARAMETER_ID: par,
        petab.PARAMETER_SCALE: petab.LIN if par.endswith(('_Ea', '_dG', '_ddG', '_phi')) else petab.LOG10,
        petab.LOWER_BOUND: lbs[par.split('_')[-1]],
        petab.UPPER_BOUND: ubs[par.split('_')[-1]],
        petab.NOMINAL_VALUE: initials[par] if par.endswith('_0') else 1.0,
        petab.ESTIMATE: False if par.endswith(('_phi', '_0')) else True,
        
    } for par in free_parameters
]).set_index(petab.PARAMETER_ID)

parameter_table


# In[20]:


import petab
import libsbml
from pysb.export import export
sbml_reader = libsbml.SBMLReader()
sbml_doc = sbml_reader.readSBMLFromString(export(model, 'sbml'))
sbml_model = sbml_doc.getModel()
# rename pysb exported observables
sbml_model.getParameter('__obs1').setId('pMEK_obs')
sbml_model.getAssignmentRuleByVariable('__obs1').setVariable('pMEK_obs')
sbml_model.getParameter('__obs2').setId('pERK_obs')
sbml_model.getAssignmentRuleByVariable('__obs2').setVariable('pERK_obs')
petab_problem = petab.Problem(
    sbml_model=sbml_model, sbml_reader=sbml_reader, sbml_document=sbml_doc,
    measurement_df=measurement_table, condition_df=condition_table,
    observable_df=observable_table, parameter_df=parameter_table
)
petab.lint.lint_problem(petab_problem)


# ## Calibrating the model in pyPESTO

# In[ ]:


import pypesto
import pypesto.petab
importer = pypesto.petab.PetabImporter(petab_problem,
                                       model_name=model.name)

pypesto_problem = importer.create_problem()


# In[ ]:


for edata in pypesto_problem.objective.edatas:
    edata.reinitializeFixedParameterInitialStates = True


# #### - switch between forward adjoint sensitivity analysis
# - compare run-time for different linear solvers
# - check accuracy of 

# In[ ]:


from pypesto.optimize import FidesOptimizer, OptimizeOptions, minimize
import fides

optimizer = FidesOptimizer(
    hessian_update=fides.HybridUpdate(),
    options={
        fides.Options.MAXTIME: 3000,
        fides.Options.SUBSPACE_DIM: fides.SubSpaceDim.TWO,
    }
)

optimize_options = OptimizeOptions(
    startpoint_resample=True, allow_failed_starts=False,
)


# In[ ]:


pypesto_problem.objective.amici_solver.setNewtonMaxSteps(0)
#pypesto_problem.objective.guess_steadystate = False


# In[ ]:


import amici
pypesto_problem.objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod.forward)
pypesto_problem.objective.amici_model.setSteadyStateSensitivityMode(amici.SteadyStateSensitivityMode.simulationFSA)
pypesto_problem.objective.amici_solver.setAbsoluteTolerance(1e-12)
pypesto_problem.objective.amici_solver.setRelativeTolerance(1e-12)


# In[ ]:


result = minimize(
    pypesto_problem, optimizer, n_starts=20, options=optimize_options,
    startpoint_method=pypesto.startpoint.uniform,
    engine=pypesto.engine.MultiThreadEngine(4)
)


# In[ ]:


from pypesto.visualize import waterfall, parameters
waterfall(result)


# In[ ]:


parameters(result)


# In[ ]:


visualization_table = pd.DataFrame([
    {
        petab.PLOT_ID: f'{rafi}_{obs}',
        petab.PLOT_TYPE_SIMULATION: petab.LINE_PLOT,
        petab.PLOT_TYPE_DATA: petab.MEAN_AND_SD,
        petab.DATASET_ID: condition,
        petab.X_VALUES: 'RAFi_0',
        petab.Y_VALUES: obs,
        petab.X_SCALE: petab.LOG10,
        petab.X_LABEL: rafi,
        petab.Y_LABEL: obs,
        petab.LEGEND_ENTRY: condition.split('__')[0]
    }
    for rafi in RAFis
    for obs in observable_table.index
    for condition in measurement_table[petab.DATASET_ID].unique()
    if condition.split('__')[1] == rafi
])


# In[ ]:


visualization_table


# In[ ]:


x = pypesto_problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                                       pypesto_problem.x_free_indices)
simulation = pypesto_problem.objective(x, return_dict=True)
simulation_df = amici.petab_objective.rdatas_to_simulation_df(
    simulation['rdatas'],
    model=importer.create_model(),
    measurement_df=measurement_table,
)


# In[ ]:


simulation_df = simulation_df.sort_values([petab.OBSERVABLE_ID, petab.PREEQUILIBRATION_CONDITION_ID, 
                                           petab.SIMULATION_CONDITION_ID]).reset_index()
measurement_table = measurement_table.sort_values([petab.OBSERVABLE_ID, petab.PREEQUILIBRATION_CONDITION_ID, 
                                                   petab.SIMULATION_CONDITION_ID]).reset_index()


# In[ ]:


import petab.visualize
petab.visualize.plot_data_and_simulation(
    exp_data=measurement_table,
    exp_conditions=condition_table,
    sim_data=simulation_df,
    vis_spec=visualization_table
);


# In[ ]:


x_dict = dict(zip(pypesto_problem.x_names, result.optimize_result.list[0]['x']))
x_dict


# In[ ]:


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pypesto_problem.objective.check_grad_multi_eps(x, multi_eps=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                                               verbosity=0)


# In[ ]:


simulation['rdatas'][0]['preeq_status']


# In[ ]:


x


# In[ ]:


pypesto_problem.x_free_indices


# In[ ]:


pypesto_problem.objective.x_names = [pypesto_problem.objective.x_names[ix] for ix in pypesto_problem.x_free_indices]


# In[ ]:


dir(pypesto_problem)


# In[ ]:




