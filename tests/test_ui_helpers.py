"""
tests/test_ui_helpers.py
─────────────────────────
Tests for sidebar navigation and session state preservation in UI helpers.
"""

from unittest.mock import MagicMock, patch
import pytest
import streamlit as st
from utils.ui_helpers import _pipeline_step, render_sidebar


def test_pipeline_step_uses_st_page_link():
    """Verify that _pipeline_step calls st.page_link instead of rendering hard HTML anchor tags."""
    with patch("streamlit.page_link") as mock_page_link, \
         patch("streamlit.markdown") as mock_markdown:
        
        _pipeline_step("4", "Quiz & Practice", done=False, active=False, page="pages/04_quiz.py")
        
        # Must call st.page_link with pages/04_quiz.py
        mock_page_link.assert_called_once_with(
            "pages/04_quiz.py",
            label="Quiz & Practice",
            icon="4️⃣"
        )
        
        # Must NOT call st.markdown with <a href
        for call_args in mock_markdown.call_args_list:
            content = str(call_args)
            assert "<a href" not in content, "Hard <a href> tag detected; must use st.page_link for session state retention!"


def test_pipeline_step_done_state_icon():
    """Verify that done steps show checkmark icon."""
    with patch("streamlit.page_link") as mock_page_link:
        _pipeline_step("1", "Upload Notes", done=True, active=False, page="pages/01_upload.py")
        mock_page_link.assert_called_once_with(
            "pages/01_upload.py",
            label="Upload Notes",
            icon="✅"
        )


def test_render_sidebar_navigation_targets():
    """Verify that all sidebar pipeline steps link to valid multipage scripts."""
    mock_sidebar = MagicMock()
    
    with patch("streamlit.sidebar", mock_sidebar), \
         patch("streamlit.page_link") as mock_page_link, \
         patch("streamlit.markdown"), \
         patch("streamlit.divider"), \
         patch("streamlit.button", return_value=False):
        
        render_sidebar(active_page="quiz")
        
        # Collect all targets passed to st.page_link
        targets = [call.args[0] for call in mock_page_link.call_args_list]
        
        assert "app.py" in targets
        assert "pages/01_upload.py" in targets
        assert "pages/02_summary.py" in targets
        assert "pages/03_keywords.py" in targets
        assert "pages/04_quiz.py" in targets
        assert "pages/05_revision.py" in targets
        assert "pages/06_nlp_analysis.py" in targets
