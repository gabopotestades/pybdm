"""Algorithms based on ``BDM`` objects."""
from itertools import product
from random import choice
import numpy as np
from networkx import from_numpy_array, number_connected_components


class PerturbationExperiment:
    """Perturbation experiment class.

    Perturbation experiment studies change of BDM / entropy under changes
    applied to the underlying dataset. This is the main tool for detecting
    parts of a system having some causal significance as opposed
    to noise parts.

    Parts which when perturbed yield negative contribution to the overall
    complexity after change are likely to be important for the system,
    since their removal make it more noisy. On the other hand parts that yield
    positive contribution to the overall complexity after change are likely
    to be noise since they elongate the system's description length.

    Attributes
    ----------
    bdm : BDM
        BDM object. It has to be configured properly to handle
        the dataset that is to be studied.
    X : array_like (optional)
        Dataset for perturbation analysis. May be set later.
    metric : {'bdm', 'ent'}
        Which metric to use for perturbing.

    See also
    --------
    pybdm.bdm.BDM : BDM computations

    Examples
    --------
    >>> import numpy as np
    >>> from pybdm import BDM, PerturbationExperiment
    >>> X = np.random.randint(0, 2, (100, 100))
    >>> bdm = BDM(ndim=2)
    >>> pe = PerturbationExperiment(bdm, metric='bdm')
    >>> pe.set_data(X)
    >>> idx = np.argwhere(X) # Perturb only ones (1 --> 0)
    >>> delta_bdm = pe.run(idx)
    >>> len(delta_bdm) == idx.shape[0]
    True

    More examples can be found in :doc:`usage`.
    """
    def __init__(self, bdm, X=None, metric='bdm'):
        """Initialization method."""
        self.bdm = bdm
        self.metric = metric
        self._counter = None
        self._value = None
        self._ncounts = None
        if self.metric == 'bdm':
            self._method = self._update_bdm
        elif self.metric == 'ent':
            self._method = self._update_ent
        else:
            raise AttributeError("Incorrect metric, not one of: 'bdm', 'ent'")
        if X is None:
            self.X = X
        else:
            self.set_data(X)

    def __repr__(self):
        cn = self.__class__.__name__
        bdm = str(self.bdm)[1:-1]
        return "<{}(metric={}) with {}>".format(cn, self.metric, bdm)

    @property
    def size(self):
        """Data size getter."""
        return self.X.size

    @property
    def shape(self):
        """Data shape getter."""
        return self.X.shape

    @property
    def ndim(self):
        """Data number of axes getter."""
        return self.X.ndim

    @property
    def subgraph_count(self):
        G = from_numpy_array(self.X)
        return number_connected_components(G)
    
    def set_data(self, X):
        """Set dataset for the perturbation experiment.

        Parameters
        ----------
        X : array_like
            Dataset to perturb.
        """
        if not np.isin(np.unique(X), range(self.bdm.nsymbols)).all():
            raise ValueError("'X' is malformed (too many or ill-mapped symbols)")
        self.X = X
        self._counter = self.bdm.decompose_and_count(X)
        if self.metric == 'bdm':
            self._value = self.bdm.compute_bdm(self._counter)
        elif self.metric == 'ent':
            self._value = self.bdm.compute_ent(self._counter)
            self._ncounts = sum(self._counter.values())
   
    def _idx_to_parts(self, idx):
        def _slice(i, k):
            start = i - i % k
            end = start + k
            return slice(start, end)
        try:
            shift = self.bdm.partition.shift
        except AttributeError:
            shift = 0
        shape = self.bdm.partition.shape
        if shift == 0:
            r_idx = tuple((k // l)*l for k, l in zip(idx, shape))
            idx = tuple(slice(k, k+l) for k, l in zip(r_idx, shape))
        else:
            idx = tuple(slice(max(0, k-l+1), k+l) for k, l in zip(idx, shape))
        yield from self.bdm.decompose(self.X[idx])

    def _update_bdm(self, idx, old_value, new_value, keep_changes):
        old_bdm = self._value
        new_bdm = self._value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter[(key, cmx)]
            if n > 1:
                new_bdm += np.log2((n-1) / n)
                if keep_changes:
                    self._counter[(key, cmx)] -= 1
            else:
                new_bdm -= cmx
                if keep_changes:
                    del self._counter[(key, cmx)]
        self.X[idx] = new_value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter.get((key, cmx), 0)
            if n > 0:
                new_bdm += np.log2((n+1) / n)
            else:
                new_bdm += cmx
            if keep_changes:
                self._counter.update([(key, cmx)])
        if not keep_changes:
            self.X[idx] = old_value
        else:
            self._value = new_bdm
        return new_bdm - old_bdm

    def _update_ent(self, idx, old_value, new_value, keep_changes):
        old_ent = self._value
        new_ent = self._value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter[(key, cmx)]
            p = n / self._ncounts
            new_ent += p*np.log2(p)
            if n > 1:
                p = (n-1) / self._ncounts
                new_ent -= p*np.log2(p)
                if keep_changes:
                    self._counter[(key, cmx)] -= 1
            elif keep_changes:
                del self._counter[(key, cmx)]
        self.X[idx] = new_value
        for key, cmx in self.bdm.lookup(self._idx_to_parts(idx)):
            n = self._counter.get((key, cmx), 0) + 1
            p = n / self._ncounts
            new_ent -= p*np.log2(p)
            if n > 1:
                p = (n-1) / self._ncounts
                new_ent += p*np.log2(p)
            if keep_changes:
                self._counter.update([(key, cmx)])
        if not keep_changes:
            self.X[idx] = old_value
        else:
            self._value = new_ent
        return new_ent - old_ent

    def perturb(self, idx, value=-1, keep_changes=False):
        """Perturb element of the dataset.

        Parameters
        ----------
        idx : tuple
            Index tuple of an element.
        value : int or callable or None
            Value to assign.
            If negative then new value is randomly selected from the set
            of other possible values.
            For binary data this is just a bit flip and no random numbers
            generation is involved in the process.
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        float :
            BDM value change.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> perturbation.perturb((10, ), -1) # doctest: +FLOAT_CMP
        26.91763012739709
        """
        old_value = self.X[idx]
        if value < 0:
            if self.bdm.nsymbols <= 2:
                value = 1 if old_value == 0 else 0
            else:
                value = choice([
                    x for x in range(self.bdm.nsymbols)
                    if x != old_value
                ])
        if old_value == value:
            return 0
        return self._method(idx, old_value, value, keep_changes)
    
    def run(self, idx=None, values=None, keep_changes=False):
        """Run perturbation experiment.

        Parameters
        ----------
        idx : array_like or None
            *Numpy* integer array providing indexes (in rows) of elements
            to perturb. If ``None`` then all elements are perturbed.
        values : array_like or None
            Value to assign during perturbation.
            Negative values correspond to changing value to other
            randomly selected symbols from the alphabet.
            If ``None`` then all values are assigned this way.
            If set then its dimensions must agree with the dimensions
            of ``idx`` (they are horizontally stacked).
        keep_changes : bool
            If ``True`` then changes in the dataset are persistent,
            so each perturbation step depends on the previous ones.

        Returns
        -------
        array_like
            1D float array with perturbation values.

        Examples
        --------
        >>> from pybdm import BDM
        >>> bdm = BDM(ndim=1)
        >>> X = np.ones((30, ), dtype=int)
        >>> perturbation = PerturbationExperiment(bdm, X)
        >>> changes = np.array([10, 20])
        >>> perturbation.run(changes) # doctest: +FLOAT_CMP
        array([26.91763013, 27.34823681])
        """
        if idx is None:
            indexes = [ range(k) for k in self.X.shape ]
            idx = np.array([ x for x in product(*indexes) ], dtype=int)
        if values is None:
            values = np.full((idx.shape[0], ), -1, dtype=int)
        return np.apply_along_axis(
            lambda r: self.perturb(tuple(r[:-1]), r[-1], keep_changes=keep_changes),
            axis=1,
            arr=np.column_stack((idx, values))
        )

    def deconvolve(self, division, keep_changes=False):
        if division >= self.X.shape[0] or self.subgraph_count >= division:
            return self.X
        
        original_idx = np.empty((0, 2), dtype=int)
        deleted_edge_graph = np.copy(self.X)

        while self.subgraph_count < division:

            info_loss = np.empty((0, 3), dtype=int)
            nonzero_edges =  np.column_stack(np.nonzero(np.triu(deleted_edge_graph)))

            for edge in nonzero_edges:
                deleted_edge_graph[edge,edge[::-1]] = 0
                deleted_edge_bdm = self.bdm.bdm(deleted_edge_graph)
                loss =  self._value - deleted_edge_bdm
                if loss > 0:
                    info_loss = np.vstack((info_loss, np.array([*edge, loss])))
            
            info_loss = info_loss[np.where(info_loss[:,-1] == info_loss[:,-1].min())]
            edges_for_deletion = info_loss[:,:-1]
            edges_for_deletion = np.array([*edges_for_deletion, *edges_for_deletion[:, ::-1]], dtype=int)
            original_idx = np.vstack((original_idx, edges_for_deletion))
            self.run(idx=edges_for_deletion,keep_changes=True)
            deleted_edge_graph = np.copy(self.X)

        if not keep_changes:
            deconvoluted_graph = np.copy(self.X)
            self.X[original_idx[:,0],original_idx[:,1]] = 1
            self.set_data(self.X)
            return deconvoluted_graph, info_loss, None, edges_for_deletion
        
        return self.X, info_loss, None, edges_for_deletion

    def deconvolve_cutoff(self, auxiliary_cutoff=None, keep_changes=False):

        info_loss = np.empty((0, 3), dtype=int)
        deleted_edge_graph = np.copy(self.X)
        nonzero_edges =  np.column_stack(np.nonzero(np.triu(deleted_edge_graph)))
        orig_bdm = self.bdm.bdm(self.X)
        
        for edge in nonzero_edges:
            deleted_edge_graph[edge,edge[::-1]] = 0
            deleted_edge_bdm = self.bdm.bdm(deleted_edge_graph)
            loss =  orig_bdm - deleted_edge_bdm
            if loss > 0:
                info_loss = np.vstack((info_loss, np.array([*edge, loss])))
            deleted_edge_graph[edge,edge[::-1]] = 1

        info_loss = info_loss[np.argsort(-info_loss[:,-1])]

        # for coord in info_loss:
        #     print(f'({int(coord[0])}, {int(coord[1])}) = {coord[2]}')

        difference = np.diff(info_loss[:, -1]) * -1

        if auxiliary_cutoff is None:
            auxiliary_cutoff = np.sqrt(
                np.sum((difference - np.log2(2))**2) / difference.shape[0]
            )
            # auxiliary_cutoff = np.std(difference)
            print(f'Auxiliary Cutoff: {auxiliary_cutoff}')
            print(f'Standard Deviation: {np.std(difference)}')

        difference_filter = [False]
        difference_filter.extend(np.isin(
            np.arange(len(difference)),
            np.where(abs(difference - np.log2(2)) > auxiliary_cutoff)
        ))
        # difference_filter = list(np.isin(
        #     np.arange(len(difference)),
        #     np.where(abs(difference - np.log2(2)) > auxiliary_cutoff)
        # ))
        # difference_filter.extend([False])

        #print(difference_filter)
        if (not any(difference_filter)):
            return self.X, None, None, None, None, None
        
        edges_for_deletion = (info_loss[difference_filter])[:, :-1]
        edges_for_deletion = np.array([*edges_for_deletion, *edges_for_deletion[:, ::-1]], dtype=int)

        if not keep_changes:
            deleted_edge_graph[edges_for_deletion[:,0],edges_for_deletion[:,1]] = 0
            return deleted_edge_graph, info_loss, difference, difference_filter, edges_for_deletion, auxiliary_cutoff

        self.run(idx=edges_for_deletion,keep_changes=keep_changes)
        return self.X, info_loss, difference, difference_filter, edges_for_deletion, auxiliary_cutoff