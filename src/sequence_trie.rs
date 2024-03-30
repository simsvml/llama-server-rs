use std::collections::hash_map::{HashMap, Entry};
use std::ops::{Index, IndexMut};
use crate::LlamaToken;


/// An index representing a node within a `SequenceTrie`.  Each node in the trie corresponds to a
/// particular sequence of tokens.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct TrieNodeId(u32);

impl TrieNodeId {
    pub const ROOT: TrieNodeId = TrieNodeId(u32::MAX);

    fn from_usize(x: usize) -> TrieNodeId {
        let x = u32::try_from(x).unwrap();
        assert!(x != u32::MAX);
        TrieNodeId(x)
    }

    fn as_usize(self) -> usize {
        self.0.try_into().unwrap()
    }
}

/// Trie for associating data with token sequences.
pub struct SequenceTrie<T> {
    /// Map from `(seq, token)` to the child produced by extending `seq` with `token`.
    child_map: HashMap<(TrieNodeId, LlamaToken), TrieNodeId>,
    /// Data associated with each node in the trie.
    data: Vec<T>,
}

impl<T> SequenceTrie<T> {
    pub fn new() -> SequenceTrie<T> {
        SequenceTrie::with_capacity(0)
    }

    pub fn with_capacity(cap: usize) -> SequenceTrie<T> {
        SequenceTrie {
            child_map: HashMap::with_capacity(cap),
            data: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.child_map.clear();
        self.data.clear();
    }

    pub fn child(&self, parent: TrieNodeId, token: LlamaToken) -> Option<TrieNodeId> {
        self.child_map.get(&(parent, token)).copied()
    }

    /// Extend the `parent` sequence with `token`, creating a new node containing `data`.  If the
    /// node already exists, its data is overwritten with `data` and its `TrieNodeId` is returned.
    /// Returns the `TrieNodeId` of the child node.
    pub fn insert(&mut self, parent: TrieNodeId, token: LlamaToken, data: T) -> TrieNodeId {
        match self.child_map.entry((parent, token)) {
            Entry::Vacant(e) => {
                let child = TrieNodeId::from_usize(self.data.len());
                e.insert(child);
                self.data.push(data);
                child
            },
            Entry::Occupied(e) => {
                let child = *e.get();
                self.data[child.as_usize()] = data;
                child
            },
        }
    }
}

impl<T> Index<TrieNodeId> for SequenceTrie<T> {
    type Output = T;
    fn index(&self, idx: TrieNodeId) -> &T {
        &self.data[idx.as_usize()]
    }
}

impl<T> IndexMut<TrieNodeId> for SequenceTrie<T> {
    fn index_mut(&mut self, idx: TrieNodeId) -> &mut T {
        &mut self.data[idx.as_usize()]
    }
}
