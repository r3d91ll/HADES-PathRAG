import unittest
from unittest.mock import MagicMock, patch
import pytest
from src.ingest.ingestor import RepositoryIngestor

class TestRepositoryIngestorEdgeCases(unittest.TestCase):
    def setUp(self):
        self.mock_processor = MagicMock()
        self.mock_storage = MagicMock()
        self.ingestor = RepositoryIngestor(config={})
        self.ingestor.processor = self.mock_processor
        self.ingestor.storage = self.mock_storage

    def test_empty_batch(self):
        # Should process nothing, return empty list
        result = self.ingestor._parallel_preprocess({})
        self.assertEqual(result, [])

    def test_invalid_batch(self):
        # Simulate processor raising exception on process_batch
        self.mock_processor.process_batch.side_effect = Exception("fail")
        # Should not raise, just log error and return empty result
        result = self.ingestor._parallel_preprocess({'bad.md': ['bad.md']})
        self.assertEqual(result, [])

    def test_ingest_document_handles_storage_error(self):
        # Simulate storage error
        self.mock_storage.store_document.side_effect = Exception("fail")
        with self.assertRaises(Exception):
            self.ingestor.ingest_document({'path': 'foo.md'})

    @patch('src.ingest.ingestor.get_pre_processor')
    def test_ingest_batch_calls_processor_and_storage(self, mock_get_pre_processor):
        # Patch get_pre_processor to return our mock_processor
        mock_get_pre_processor.return_value = self.mock_processor
        self.mock_processor.process_batch.return_value = [{'path': 'foo.md'}]
        self.mock_storage.store_document.return_value = None
        batch = {'foo.md': ['foo.md']}
        self.ingestor._parallel_preprocess(batch)
        self.mock_processor.process_batch.assert_called_with(['foo.md'])

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    @patch('src.ingest.ingestor.RepositoryIngestor._parallel_preprocess')
    @patch('src.ingest.ingestor.RepositoryIngestor._extract_relationships')
    @patch('src.ingest.ingestor.RepositoryIngestor._generate_embeddings')
    @patch('src.ingest.ingestor.RepositoryIngestor._store_documents')
    def test_ingest_happy_path(self, mock_store, mock_embed, mock_extract, mock_preprocess, mock_batcher, mock_setup):
        # Setup mocks for a successful ingestion
        mock_batcher.return_value.collect_files.return_value = {'py': ['foo.py']}
        mock_batcher.return_value.get_stats.return_value = {'total': 1}
        mock_preprocess.return_value = [{'path': 'foo.py'}]
        mock_extract.return_value = [{'from': 'a', 'to': 'b'}]
        mock_embed.return_value = [{'path': 'foo.py', 'embedding': [0.1, 0.2]}]
        mock_store.return_value = {'stored': 1}
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        stats = ingestor.ingest('/tmp', dataset_name='test')
        self.assertEqual(stats['file_stats']['total'], 1)
        self.assertEqual(stats['document_count'], 1)
        self.assertEqual(stats['relationship_count'], 1)
        self.assertEqual(stats['storage_stats']['stored'], 1)

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    def test_ingest_empty_directory(self, mock_batcher, mock_setup):
        mock_batcher.return_value.collect_files.return_value = {}
        mock_batcher.return_value.get_stats.return_value = {'total': 0}
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        with patch.object(ingestor, '_parallel_preprocess', return_value=[]), \
             patch.object(ingestor, '_extract_relationships', return_value=[]), \
             patch.object(ingestor, '_generate_embeddings', return_value=[]), \
             patch.object(ingestor, '_store_documents', return_value={}):
            stats = ingestor.ingest('/tmp', dataset_name='empty')
            self.assertEqual(stats['file_stats']['total'], 0)
            self.assertEqual(stats['document_count'], 0)
            self.assertEqual(stats['relationship_count'], 0)

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    def test_ingest_directory_permission_error(self, mock_batcher, mock_setup):
        mock_batcher.return_value.collect_files.side_effect = PermissionError("No permission")
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        with self.assertRaises(PermissionError):
            ingestor.ingest('/restricted', dataset_name='fail')

    def test_ingest_invalid_config(self):
        # Simulate invalid config missing database keys
        bad_config = {'database': {'host': None, 'username': None}}
        with patch('src.ingest.ingestor.ArangoConnection') as mock_conn:
            RepositoryIngestor(config=bad_config)
            mock_conn.assert_called()

    @patch('src.ingest.ingestor.ArangoConnection')
    def test_ingest_db_connection_failure(self, mock_conn):
        mock_conn.side_effect = Exception("DB fail")
        with self.assertRaises(Exception):
            RepositoryIngestor(config={})

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    def test_ingest_batcher_failure(self, mock_batcher, mock_setup):
        mock_batcher.return_value.collect_files.side_effect = Exception("batch fail")
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        with self.assertRaises(Exception):
            ingestor.ingest('/tmp', dataset_name='batchfail')

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    def test_ingest_parallel_preprocess_error(self, mock_batcher, mock_setup):
        mock_batcher.return_value.collect_files.return_value = {'py': ['foo.py']}
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        with patch.object(ingestor, '_parallel_preprocess', side_effect=Exception("preprocess fail")):
            with self.assertRaises(Exception):
                ingestor.ingest('/tmp', dataset_name='preprocessfail')

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    def test_ingest_generate_embeddings_error(self, mock_batcher, mock_setup):
        mock_batcher.return_value.collect_files.return_value = {'py': ['foo.py']}
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        with patch.object(ingestor, '_parallel_preprocess', return_value=[{'path': 'foo.py'}]), \
             patch.object(ingestor, '_extract_relationships', return_value=[]), \
             patch.object(ingestor, '_generate_embeddings', side_effect=Exception("embedding fail")):
            with self.assertRaises(Exception):
                ingestor.ingest('/tmp', dataset_name='embedfail')

    @patch('src.ingest.ingestor.RepositoryIngestor.setup_collections')
    @patch('src.ingest.ingestor.FileBatcher')
    def test_ingest_storage_failure(self, mock_batcher, mock_setup):
        mock_batcher.return_value.collect_files.return_value = {'py': ['foo.py']}
        ingestor = RepositoryIngestor(config={})
        ingestor.batcher = mock_batcher.return_value
        with patch.object(ingestor, '_parallel_preprocess', return_value=[{'path': 'foo.py'}]), \
             patch.object(ingestor, '_extract_relationships', return_value=[]), \
             patch.object(ingestor, '_generate_embeddings', return_value=[{'path': 'foo.py', 'embedding': [0.1]}]), \
             patch.object(ingestor, '_store_documents', side_effect=Exception("store fail")):
            with self.assertRaises(Exception):
                ingestor.ingest('/tmp', dataset_name='storefail')

class TestRepositoryIngestorInternals(unittest.TestCase):
    def setUp(self):
        self.ingestor = RepositoryIngestor(config={})
        self.ingestor.db_connection = MagicMock()
        self.ingestor.CODE_NODE_COLLECTION = "code_nodes"
        self.ingestor.CODE_EDGE_COLLECTION = "code_edges"
        self.ingestor.CODE_GRAPH_NAME = "code_graph"

    @patch('src.ingest.ingestor.get_pre_processor')
    def test_parallel_preprocess_happy_path(self, mock_get_pre_processor):
        mock_proc = MagicMock()
        mock_proc.process_batch.return_value = [{"id": "a", "type": "py"}]
        mock_get_pre_processor.return_value = mock_proc
        batches = {"py": ["a.py"]}
        result = self.ingestor._parallel_preprocess(batches)
        self.assertEqual(result, [{"id": "a", "type": "py"}])

    @patch('src.ingest.ingestor.get_pre_processor')
    def test_parallel_preprocess_skips_unsupported(self, mock_get_pre_processor):
        mock_get_pre_processor.side_effect = ValueError("unsupported")
        batches = {"unknown": ["a.unknown"]}
        result = self.ingestor._parallel_preprocess(batches)
        self.assertEqual(result, [])

    @patch('src.ingest.ingestor.get_pre_processor')
    def test_parallel_preprocess_handles_batch_error(self, mock_get_pre_processor):
        mock_proc = MagicMock()
        mock_proc.process_batch.side_effect = Exception("fail")
        mock_get_pre_processor.return_value = mock_proc
        batches = {"py": ["a.py"]}
        result = self.ingestor._parallel_preprocess(batches)
        self.assertEqual(result, [])

    def test_extract_relationships_deduplication(self):
        docs = [
            {"relationships": [
                {"from": "a", "to": "b", "type": "CALLS"},
                {"from": "a", "to": "b", "type": "CALLS"},
                {"from": "b", "to": "c", "type": "CONTAINS"}
            ]}
        ]
        rels = self.ingestor._extract_relationships(docs)
        self.assertEqual(len(rels), 2)
        self.assertTrue(any(r["type"] == "CALLS" for r in rels))
        self.assertTrue(any(r["type"] == "CONTAINS" for r in rels))

    def test_extract_relationships_empty(self):
        docs = [{"id": "a"}, {"id": "b"}]
        rels = self.ingestor._extract_relationships(docs)
        self.assertEqual(rels, [])

    def test_store_documents_happy_path(self):
        self.ingestor.db_connection.insert_document = MagicMock()
        self.ingestor.db_connection.insert_edge = MagicMock()
        docs = [{"id": "a", "type": "py"}]
        rels = [{"from": "a", "to": "b", "type": "CALLS"}]
        stats = self.ingestor._store_documents(docs, rels)
        self.assertEqual(stats["node_count"], 1)
        self.assertEqual(stats["edge_count"], 1)

    def test_store_documents_handles_doc_error(self):
        self.ingestor.db_connection.insert_document = MagicMock(side_effect=Exception("fail"))
        self.ingestor.db_connection.insert_edge = MagicMock()
        docs = [{"id": "a", "type": "py"}]
        rels = []
        stats = self.ingestor._store_documents(docs, rels)
        self.assertEqual(stats["node_count"], 0)

    def test_store_documents_handles_edge_error(self):
        self.ingestor.db_connection.insert_document = MagicMock()
        self.ingestor.db_connection.insert_edge = MagicMock(side_effect=Exception("fail"))
        docs = [{"id": "a", "type": "py"}]
        rels = [{"from": "a", "to": "b", "type": "CALLS"}]
        stats = self.ingestor._store_documents(docs, rels)
        self.assertEqual(stats["edge_count"], 0)

    def test_normalize_key_replaces_invalid_chars(self):
        key = "foo/bar.baz:qux quux"
        norm = RepositoryIngestor._normalize_key(key)
        self.assertNotIn("/", norm)
        self.assertNotIn(".", norm)
        self.assertNotIn(":", norm)
        self.assertNotIn(" ", norm)
        self.assertNotIn("\\", norm)

    def test_setup_collections_creates_all(self):
        db = self.ingestor.db_connection
        db.graph_exists.return_value = False
        db.collection_exists.side_effect = [False, False]
        self.ingestor.setup_collections()
        db.create_graph.assert_called()
        db.create_collection.assert_called()
        db.create_edge_collection.assert_called()

    def test_setup_collections_skips_existing(self):
        db = self.ingestor.db_connection
        db.graph_exists.return_value = True
        db.collection_exists.side_effect = [True, True]
        self.ingestor.setup_collections()
        db.create_graph.assert_not_called()
        db.create_collection.assert_not_called()
        db.create_edge_collection.assert_not_called()

    def test_setup_collections_handles_error(self):
        db = self.ingestor.db_connection
        db.graph_exists.side_effect = Exception("fail")
        with self.assertRaises(Exception):
            self.ingestor.setup_collections()

class TestRepositoryIngestorDeepInternals(unittest.TestCase):
    def setUp(self):
        self.ingestor = RepositoryIngestor(config={})
        self.ingestor.db_connection = MagicMock()
        self.ingestor.CODE_NODE_COLLECTION = "code_nodes"
        self.ingestor.CODE_EDGE_COLLECTION = "code_edges"
        self.ingestor.CODE_GRAPH_NAME = "code_graph"
        self.ingestor.NODE_TYPES = {"REPOSITORY": "repo", "FILE": "file", "MODULE": "module", "FUNCTION": "function", "CLASS": "class", "METHOD": "method"}
        self.ingestor.EDGE_TYPES = {"CONTAINS": "CONTAINS"}

    @patch('src.ingest.isne_connector.ISNEIngestorConnector')
    def test_generate_embeddings_success(self, mock_isne):
        mock_conn = mock_isne.return_value
        mock_conn.process_documents.return_value = [{"id": "a", "embedding": [1,2,3]}]
        docs = [{"id": "a"}]
        result = self.ingestor._generate_embeddings(docs)
        self.assertEqual(result, [{"id": "a", "embedding": [1,2,3]}])

    @patch('src.ingest.isne_connector.ISNEIngestorConnector')
    def test_generate_embeddings_fallback_on_error(self, mock_isne):
        mock_conn = mock_isne.return_value
        mock_conn.process_documents.side_effect = Exception("fail")
        docs = [{"id": "a"}]
        result = self.ingestor._generate_embeddings(docs)
        self.assertEqual(result, docs)

    def test_create_repo_node(self):
        self.ingestor.db_connection.insert_document.return_value = {"_key": "repo"}
        repo_info = {
            "repo_name": "MyRepo",
            "remote_url": "url",
            "branches": ["main"],
            "current_branch": "main",
            "commit_count": 1,
            "last_commit": "abc",
            "contributors": ["alice"]
        }
        out = self.ingestor._create_repo_node(repo_info)
        self.assertEqual(out["_key"], "repo")
        self.ingestor.db_connection.insert_document.assert_called()

    def test_update_repo_node(self):
        self.ingestor.db_connection.update_document = MagicMock()
        stats = {"nodes_created": 1, "edges_created": 2, "files_processed": 3}
        self.ingestor._update_repo_node("myrepo", stats)
        self.ingestor.db_connection.update_document.assert_called_with(
            "code_nodes", "myrepo", unittest.mock.ANY
        )

    def test_process_code_files_minimal(self):
        self.ingestor.db_connection.insert_document = MagicMock(return_value={"_id": "id", "_key": "key"})
        self.ingestor._create_edge = MagicMock()
        class Dummy:
            def __init__(self):
                self.code = "code"
                self.docstring = "doc"
                self.imports = []
                self.functions = {}
                self.classes = {}
                self.name = "mod"
        modules = {"foo.py": Dummy()}
        out = self.ingestor._process_code_files(modules, "repo")
        self.assertIn("foo.py", out)
        self.ingestor.db_connection.insert_document.assert_called()
        self.ingestor._create_edge.assert_called()

    def test_process_code_files_with_functions_and_classes(self):
        self.ingestor.db_connection.insert_document = MagicMock(return_value={"_id": "id", "_key": "key"})
        self.ingestor._create_edge = MagicMock()
        class DummyFunc:
            def __init__(self):
                self.docstring = "doc"
                self.code = "code"
                self.parameters = []
                self.return_type = "int"
                self.function_calls = []
                self.line_start = 1
                self.line_end = 2
        class DummyClass:
            def __init__(self):
                self.docstring = "doc"
                self.code = "code"
                self.base_classes = []
                self.line_start = 1
                self.line_end = 2
                self.methods = {"meth": DummyFunc()}
        class DummyMod:
            def __init__(self):
                self.code = "code"
                self.docstring = "doc"
                self.imports = []
                self.functions = {"func": DummyFunc()}
                self.classes = {"cls": DummyClass()}
                self.name = "mod"
        modules = {"foo.py": DummyMod()}
        out = self.ingestor._process_code_files(modules, "repo")
        self.assertIn("foo.py", out)
        self.ingestor.db_connection.insert_document.assert_called()
        self.ingestor._create_edge.assert_called()

class TestRepositoryIngestorEdgeInternals(unittest.TestCase):
    def setUp(self):
        self.ingestor = RepositoryIngestor(config={})
        self.ingestor.db_connection = MagicMock()
        self.ingestor.CODE_NODE_COLLECTION = "code_nodes"
        self.ingestor.CODE_EDGE_COLLECTION = "code_edges"
        self.ingestor.CODE_GRAPH_NAME = "code_graph"
        self.ingestor.NODE_TYPES = {"DOCUMENTATION": "doc", "DOC_SECTION": "section"}
        self.ingestor.EDGE_TYPES = {"CONTAINS": "CONTAINS", "IMPORTS": "IMPORTS", "INHERITS": "INHERITS", "CALLS": "CALLS", "DOCUMENTS": "DOCUMENTS"}

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="file content")
    def test_process_doc_files_minimal(self, mock_open):
        self.ingestor.db_connection.insert_document = MagicMock(return_value={"_id": "id", "_key": "key"})
        self.ingestor._create_edge = MagicMock()
        class DummyDocElem:
            def __init__(self):
                self.title = "section"
                self.section_type = "header"
                self.content = "section content"
                self.references = []
                self.line_start = 1
                self.line_end = 2
        class DummyDoc:
            def __init__(self):
                self.file_path = "README.md"
                self.elements = [DummyDocElem()]
        doc_files = {"README.md": DummyDoc()}
        out = self.ingestor._process_doc_files(doc_files, "repo")
        self.assertIn("README.md", out)
        self.ingestor.db_connection.insert_document.assert_called()
        self.ingestor._create_edge.assert_called()

    def test_create_code_relationships_imports(self):
        self.ingestor._create_edge = MagicMock()
        rels = {"imports": [{"source": "a.py", "target": "b.py", "alias": "b", "weight": 0.7}]}
        file_nodes = {"a.py": {"id": "1", "key": "a"}, "b.py": {"id": "2", "key": "b"}}
        count = self.ingestor._create_code_relationships(rels, file_nodes)
        self.assertEqual(count, 1)
        self.ingestor._create_edge.assert_called_with(from_id="1", to_id="2", edge_type="IMPORTS", weight=0.7, attributes={"alias": "b"})

    def test_create_code_relationships_inherits_and_calls(self):
        self.ingestor._create_edge = MagicMock()
        rels = {
            "inherits": [{"source": "a.py::A", "target": "b.py::B", "weight": 0.8}],
            "calls": [{"source": "a.py::foo", "target": "b.py::bar", "weight": 0.6}]
        }
        file_nodes = {"a.py": {"id": "1", "key": "a"}, "b.py": {"id": "2", "key": "b"}}
        count = self.ingestor._create_code_relationships(rels, file_nodes)
        self.assertEqual(count, 2)
        # Should call _create_edge with from_key/to_key for inherits/calls
        self.assertTrue(any(
            call.kwargs.get("edge_type") == "INHERITS" or call.kwargs.get("edge_type") == "CALLS"
            for call in self.ingestor._create_edge.mock_calls
        ))

    def test_create_code_relationships_handles_edge_error(self):
        def raise_err(*a, **kw):
            raise Exception("fail")
        self.ingestor._create_edge = MagicMock(side_effect=raise_err)
        rels = {"inherits": [{"source": "a.py::A", "target": "b.py::B"}]}
        file_nodes = {"a.py": {"id": "1", "key": "a"}, "b.py": {"id": "2", "key": "b"}}
        count = self.ingestor._create_code_relationships(rels, file_nodes)
        self.assertEqual(count, 0)

    def test_create_doc_code_relationships_success(self):
        self.ingestor._create_edge = MagicMock()
        rels = [{"source": "doc.md::Intro", "target": "code.py", "weight": 0.8}]
        file_nodes = {"code.py": {"id": "2", "key": "b"}}
        doc_nodes = {"doc.md": {"id": "1", "key": "a"}}
        count = self.ingestor._create_doc_code_relationships(rels, file_nodes, doc_nodes)
        self.assertEqual(count, 1)
        self.ingestor._create_edge.assert_called_with(
            from_id="1", to_id="2", edge_type="DOCUMENTS", weight=0.8, attributes={"section_title": "Intro"}
        )

    def test_create_doc_code_relationships_handles_edge_error(self):
        def raise_err(*a, **kw):
            raise Exception("fail")
        self.ingestor._create_edge = MagicMock(side_effect=raise_err)
        rels = [{"source": "doc.md::Intro", "target": "code.py"}]
        file_nodes = {"code.py": {"id": "2", "key": "b"}}
        doc_nodes = {"doc.md": {"id": "1", "key": "a"}}
        count = self.ingestor._create_doc_code_relationships(rels, file_nodes, doc_nodes)
        self.assertEqual(count, 0)

    def test_create_doc_code_relationships_unmatched_target(self):
        self.ingestor._create_edge = MagicMock()
        rels = [{"source": "doc.md::Intro", "target": "missing.py"}]
        file_nodes = {"code.py": {"id": "2", "key": "b"}}
        doc_nodes = {"doc.md": {"id": "1", "key": "a"}}
        count = self.ingestor._create_doc_code_relationships(rels, file_nodes, doc_nodes)
        self.assertEqual(count, 0)

    def test_create_edge_id_and_key_variants(self):
        self.ingestor.db_connection.insert_edge = MagicMock(return_value={"_key": "e"})
        # Using from_id/to_id
        out = self.ingestor._create_edge(from_id="id1", to_id="id2", edge_type="CALLS", weight=0.5)
        self.assertEqual(out["_key"], "e")
        # Using from_key/to_key
        out2 = self.ingestor._create_edge(from_key="k1", to_key="k2", edge_type="IMPORTS", weight=0.7)
        self.assertEqual(out2["_key"], "e")
        # Error if neither
        with self.assertRaises(ValueError):
            self.ingestor._create_edge(edge_type="CALLS")

class TestRepositoryIngestorPublicMethods(unittest.TestCase):
    def setUp(self):
        self.ingestor = RepositoryIngestor(config={})
        self.ingestor.setup_collections = MagicMock()
        self.ingestor._create_repo_node = MagicMock(return_value={"_key": "repo"})
        self.ingestor._process_code_files = MagicMock(return_value={"f.py": {"id": "id1", "key": "k1"}})
        self.ingestor._process_doc_files = MagicMock(return_value={"README.md": {"id": "id2", "key": "k2"}})
        self.ingestor._create_code_relationships = MagicMock(return_value=1)
        self.ingestor._create_doc_code_relationships = MagicMock(return_value=1)
        self.ingestor._update_repo_node = MagicMock()

    @patch('src.ingest.ingestor.GitOperations')
    @patch('src.ingest.ingestor.DocParser')
    @patch('src.ingest.ingestor.CodeParser')
    def test_ingest_repository_happy_path(self, mock_code_parser, mock_doc_parser, mock_git_ops):
        mock_git = mock_git_ops.return_value
        mock_git.clone_repository.return_value = (True, "cloned", "/tmp/repo")
        mock_git.get_repo_info.return_value = {"repo_name": "repo", "remote_url": "url", "branches": [], "current_branch": "main", "commit_count": 1, "last_commit": "abc", "contributors": []}
        mock_code = mock_code_parser.return_value
        mock_code.parse_repository.return_value = {"f.py": object()}
        mock_code.extract_relationships.return_value = {"imports": []}
        mock_doc = mock_doc_parser.return_value
        mock_doc.parse_documentation.return_value = {"README.md": object()}
        mock_doc.extract_doc_code_relationships.return_value = []
        success, msg, stats = self.ingestor.ingest_repository("http://repo")
        self.assertTrue(success)
        self.assertIn("Successfully ingested", msg)
        self.assertIn("repo_url", stats)
        self.ingestor.setup_collections.assert_called()
        self.ingestor._create_repo_node.assert_called()
        self.ingestor._process_code_files.assert_called()
        self.ingestor._process_doc_files.assert_called()
        self.ingestor._create_code_relationships.assert_called()
        self.ingestor._create_doc_code_relationships.assert_called()
        self.ingestor._update_repo_node.assert_called()

    @patch('src.ingest.ingestor.GitOperations')
    def test_ingest_repository_clone_failure(self, mock_git_ops):
        mock_git = mock_git_ops.return_value
        mock_git.clone_repository.return_value = (False, "fail", None)
        success, msg, stats = self.ingestor.ingest_repository("http://repo")
        self.assertFalse(success)
        # Check that the error message contains 'fail'
        self.assertEqual(msg, "fail")
        self.assertIn("Failed to clone", stats["errors"][0])

    @patch('src.utils.git_operations.GitOperations')
    def test_ingest_repository_exception(self, mock_git_ops):
        self.ingestor.setup_collections.side_effect = Exception("failsetup")
        success, msg, stats = self.ingestor.ingest_repository("http://repo")
        self.assertFalse(success)
        self.assertIn("failsetup", msg)
        self.assertIn("Error ingesting repository", stats["errors"][0])

    def test_process_repository_with_isne_happy_path(self):
        mock_dataset = MagicMock()
        mock_dataset.documents = [1,2]
        mock_dataset.relations = [3]
        mock_dataset.id = "dsid"
        mock_dataset.name = "dsname"
        self.ingestor._isne_connector = MagicMock()
        self.ingestor._isne_connector.process_repository.return_value = mock_dataset
        stats = self.ingestor.process_repository_with_isne("/tmp/repo", "repo")
        self.assertEqual(stats["repository_name"], "repo")
        self.assertEqual(stats["pipeline"], "isne")
        self.assertEqual(stats["document_count"], 2)
        self.assertEqual(stats["relation_count"], 1)
        self.assertEqual(stats["dataset_id"], "dsid")
        self.assertEqual(stats["dataset_name"], "dsname")

    def test_process_repository_with_isne_none_dataset(self):
        self.ingestor._isne_connector = MagicMock()
        self.ingestor._isne_connector.process_repository.return_value = None
        stats = self.ingestor.process_repository_with_isne("/tmp/repo", "repo")
        self.assertEqual(stats["repository_name"], "repo")
        self.assertEqual(stats["pipeline"], "isne")
        self.assertNotIn("document_count", stats)

    def test_process_repository_with_isne_exception(self):
        self.ingestor._isne_connector = MagicMock()
        self.ingestor._isne_connector.process_repository.side_effect = Exception("failisne")
        # Patch logger to suppress error output
        with patch('src.ingest.ingestor.logger') as mock_logger:
            with self.assertRaises(Exception):
                self.ingestor.process_repository_with_isne("/tmp/repo", "repo")
            mock_logger.error.assert_not_called()  # Not caught in method

if __name__ == '__main__':
    unittest.main()
