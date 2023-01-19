import os
import logging
from itertools import chain
from urllib.parse import urlparse
from typing import List, Dict, Any

import numpy as np
from AnyQt.QtWidgets import \
    QStyle, QComboBox, QMessageBox, QGridLayout, QLabel, \
    QLineEdit, QSizePolicy as Policy, QCompleter
from AnyQt.QtCore import Qt, QTimer, QSize, QUrl
from AnyQt.QtGui import QBrush

from orangewidget.utils.filedialogs import format_filter
from orangewidget.workflow.drophandler import SingleUrlDropHandler

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from Orange.data.table import Table,StringVariable, get_sample_datasets_dir
from Orange.data.io import FileFormat, UrlReader, class_from_qualified_name
from Orange.data.io_base import MissingReaderException
from Orange.util import log_warnings
from Orange.widgets import widget, gui
from Orange.widgets.utils.localization import pl
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler, SettingProvider
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.filedialogs import RecentPathsWComboMixin, \
    open_filename_dialog, stored_recent_paths_prepend
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output, Msg

class LineEditSelectOnFocus(QLineEdit):
    def focusInEvent(self, event):
        super().focusInEvent(event)
        # If selectAll is called directly, placing the cursor unselects the text
        QTimer.singleShot(0, self.selectAll)
class MyWidget(widget.OWWidget, RecentPathsWComboMixin):
    name = "Set Workfolder"
    description = "Set the workfolder where the data is going to be saved"
    icon = "icons/mywidget.svg"
    priority = 10
    category = "Data"
    keywords = ["folder", "load", "read", "open"]

    class Outputs:
        data = Output("Data", StringVariable)

    want_main_area = False
    buttons_area_orientation = None

    SIZE_LIMIT = 1e7
    LOCAL_FILE, URL = range(2)

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    recent_urls = Setting([])
    source = Setting(LOCAL_FILE)
    url = Setting("")

    variables = ContextSetting([])

    domain_editor = SettingProvider(DomainEditor)

    class Information(widget.OWWidget.Information):
        no_file_selected = Msg("No file selected.")

    class Warning(widget.OWWidget.Warning):
        file_too_big = Msg("The file is too large to load automatically."
                           " Press Reload to load.")
        load_warning = Msg("Read warning:\n{}")
        performance_warning = Msg(
            "Categorical variables with >100 values may decrease performance.")
        renamed_vars = Msg("Some variables have been renamed "
                           "to avoid duplicates.\n{}")
        multiple_targets = Msg("Most widgets do not support multiple targets")

    class Error(widget.OWWidget.Error):
        file_not_found = Msg("File not found.")
        missing_reader = Msg("Missing reader.")
        sheet_error = Msg("Error listing available sheets.")
        unknown = Msg("Read error:\n{}")


    def __init__(self):
        super().__init__()
        RecentPathsWComboMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.available_readers = [".py"]

        layout = QGridLayout()
        layout.setSpacing(4)
        gui.widgetBox(self.controlArea, orientation=layout, box='Source')
        vbox = gui.radioButtons(None, self, "source", box=True,
                                callback=self.load_data, addToLayout=False)

        rb_button = gui.appendRadioButton(vbox, "File:", addToLayout=False)
        layout.addWidget(rb_button, 0, 0, Qt.AlignVCenter)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.file_combo.setSizePolicy(Policy.Expanding, Policy.Fixed)
        self.file_combo.setMinimumSize(QSize(100, 1))
        self.file_combo.activated[int].connect(self.select_file)
        box.layout().addWidget(self.file_combo)
        layout.addWidget(box, 0, 1)

        file_button = gui.button(
            None, self, '...', callback=self.browse_file, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.load_data, autoDefault=False)
        reload_button.setIcon(self.style().standardIcon(
            QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 3)


        rb_button = gui.appendRadioButton(vbox, "URL:", addToLayout=False)
        layout.addWidget(rb_button, 3, 0, Qt.AlignVCenter)

        self.url_combo = url_combo = QComboBox()
        url_combo.setLineEdit(LineEditSelectOnFocus())
        url_combo.setSizePolicy(Policy.Ignored, Policy.Fixed)
        url_combo.setEditable(True)
        url_combo.setInsertPolicy(url_combo.InsertAtTop)
        url_edit = url_combo.lineEdit()
        margins = url_edit.textMargins()
        l, t, r, b = margins.left(), margins.top(), margins.right(), margins.bottom()
        url_edit.setTextMargins(l + 5, t, r, b)
        layout.addWidget(url_combo, 3, 1, 1, 3)
        url_combo.activated.connect(self._url_set)
        # whit completer we set that combo box is case sensitive when
        # matching the history
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseSensitive)
        url_combo.setCompleter(completer)


        box = gui.vBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(box, 'No data loaded.')

        box = gui.widgetBox(self.controlArea, "Columns (Double click to edit)")
        self.domain_editor = DomainEditor(self)
        self.editor_model = self.domain_editor.model()
        box.layout().addWidget(self.domain_editor)

        box = gui.hBox(box)
        gui.button(
            box, self, "Reset", callback=self.reset_domain_edit,
            autoDefault=False
        )
        gui.rubber(box)
        self.apply_button = gui.button(
            box, self, "Apply", callback=self.apply_domain_edit)
        self.apply_button.setEnabled(False)
        self.apply_button.setFixedWidth(170)
        self.editor_model.dataChanged.connect(
            lambda: self.apply_button.setEnabled(True))

        hBox = gui.hBox(self.controlArea)
        gui.rubber(hBox)
        gui.button(
            hBox, self, "Browse recent datasets",
            callback=lambda: self.browse_file(True), autoDefault=False)
        gui.rubber(hBox)

        self.set_file_list()
        # Must not call open_file from within __init__. open_file
        # explicitly re-enters the event loop (by a progress bar)

        self.setAcceptDrops(True)

        if self.source == self.LOCAL_FILE:
            last_path = self.last_path()
            if last_path and os.path.exists(last_path) and \
                    os.path.getsize(last_path) > self.SIZE_LIMIT:
                self.Warning.file_too_big()
                return

        QTimer.singleShot(0, self.load_data)

    @staticmethod
    def sizeHint():
        return QSize(600, 550)

    def select_file(self, n):
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.source = self.LOCAL_FILE

            self.set_file_list()


    def _url_set(self):
        url = self.url_combo.currentText()
        pos = self.recent_urls.index(url)
        url = url.strip()

        if not urlparse(url).scheme:
            url = 'http://' + url
            self.url_combo.setItemText(pos, url)
            self.recent_urls[pos] = url

        self.source = self.URL
        self.load_data()

    def browse_file(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation datasets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        filename, reader, _ = open_filename_dialog(start_file, None, self.available_readers)
        if not filename:
            return
        self.add_path(filename)
        if reader is not None:
            self.recent_paths[0].file_format = reader.qualified_name()

        self.source = self.LOCAL_FILE
        self.load_data()

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        # We need to catch any exception type since anything can happen in
        # file readers
        self.closeContext()
        self.domain_editor.set_domain(None)
        self.apply_button.setEnabled(False)
        self.clear_messages()
        self.set_file_list()

        error = self._try_load()
        if error:
            error()
            self.data = None
            self.sheet_box.hide()
            self.Outputs.data.send(None)
            self.infolabel.setText("No data.")

    def _try_load(self):
        self._initialize_reader_combo()

        # pylint: disable=broad-except
        if self.source == self.LOCAL_FILE:
            if self.last_path() is None:
                return self.Information.no_file_selected
            elif not os.path.exists(self.last_path()):
                return self.Error.file_not_found
        else:
            url = self.url_combo.currentText().strip()
            if not url:
                return self.Information.no_file_selected

        def mark_problematic_reader():
            self.reader_combo.setItemData(self.reader_combo.currentIndex(),
                                          QBrush(Qt.red), Qt.ForegroundRole)

        


        self.loaded_file = self.last_path()
        self.data = url
        self.apply_domain_edit()  # sends data
        return None

    def _get_reader(self) -> FileFormat:
        if self.source == self.LOCAL_FILE:
            path = self.last_path()
            self.reader_combo.setEnabled(True)
            if self.recent_paths and self.recent_paths[0].file_format:
                qname = self.recent_paths[0].file_format
                qname_index = {r.qualified_name(): i for i, r in enumerate(self.available_readers)}
                if qname in qname_index:
                    self.reader_combo.setCurrentIndex(qname_index[qname] + 1)
                else:
                    # reader may be accessible, but not in self.available_readers
                    # (perhaps its code was moved)
                    self.reader_combo.addItem(qname)
                    self.reader_combo.setCurrentIndex(len(self.reader_combo) - 1)
                try:
                    reader_class = class_from_qualified_name(qname)
                except Exception as ex:
                    raise MissingReaderException(f'Can not find reader "{qname}"') from ex
                reader = reader_class(path)
            else:
                self.reader_combo.setCurrentIndex(0)
                reader = FileFormat.get_reader(path)
            if self.recent_paths and self.recent_paths[0].sheet:
                reader.select_sheet(self.recent_paths[0].sheet)
            return reader
        else:
            url = self.url_combo.currentText().strip()
            return UrlReader(url)

    def _update_sheet_combo(self):
        if len(self.reader.sheets) < 2:
            self.sheet_box.hide()
            self.reader.select_sheet(None)
            return

        self.sheet_combo.clear()
        self.sheet_combo.addItems(self.reader.sheets)
        self._select_active_sheet()
        self.sheet_box.show()

    def _select_active_sheet(self):
        try:
            idx = self.reader.sheets.index(self.reader.sheet)
            self.sheet_combo.setCurrentIndex(idx)
        except ValueError:
            # Requested sheet does not exist in this file
            self.reader.select_sheet(None)
            self.sheet_combo.setCurrentIndex(0)

        # additional readers may be added in self._get_reader()

    @staticmethod
    def _describe(table):
        def missing_prop(prop):
            if prop:
                return f"({prop * 100:.1f}% missing values)"
            else:
                return "(no missing values)"

        domain = table.domain
        text = ""

        attrs = getattr(table, "attributes", {})
        descs = [attrs[desc]
                 for desc in ("Name", "Description") if desc in attrs]
        if len(descs) == 2:
            descs[0] = f"<b>{descs[0]}</b>"
        if descs:
            text += f"<p>{'<br/>'.join(descs)}</p>"

        text += f"<p>{len(table)} {pl(len(table), 'instance')}"

        missing_in_attr = missing_in_class = ""
        if table.X.size < MyWidget.SIZE_LIMIT:
            missing_in_attr = missing_prop(table.get_nan_frequency_attribute())
            missing_in_class = missing_prop(table.get_nan_frequency_class())
        nattrs = len(domain.attributes)
        text += f"<br/>{nattrs} {pl(nattrs, 'feature')} {missing_in_attr}"
        if domain.has_continuous_class:
            text += f"<br/>Regression; numerical class {missing_in_class}"
        elif domain.has_discrete_class:
            nvals = len(domain.class_var.values)
            text += "<br/>Classification; categorical class " \
                f"with {nvals} {pl(nvals, 'value')} {missing_in_class}"
        elif table.domain.class_vars:
            ntargets = len(table.domain.class_vars)
            text += "<br/>Multi-target; " \
                f"{ntargets} target {pl(ntargets, 'variable')} " \
                f"{missing_in_class}"
        else:
            text += "<br/>Data has no target variable."
        nmetas = len(domain.metas)
        text += f"<br/>{nmetas} {pl(nmetas, 'meta attribute')}"
        text += "</p>"

        if 'Timestamp' in table.domain:
            # Google Forms uses this header to timestamp responses
            text += f"<p>First entry: {table[0, 'Timestamp']}<br/>" \
                f"Last entry: {table[-1, 'Timestamp']}</p>"
        return text

    def storeSpecificSettings(self):
        self.current_context.modified_variables = self.variables[:]

    def retrieveSpecificSettings(self):
        if hasattr(self.current_context, "modified_variables"):
            self.variables[:] = self.current_context.modified_variables

    def reset_domain_edit(self):
        self.domain_editor.reset_domain()
        self.apply_domain_edit()

    def _inspect_discrete_variables(self, domain):
        for var in chain(domain.variables, domain.metas):
            if var.is_discrete and len(var.values) > 100:
                self.Warning.performance_warning()

    def apply_domain_edit(self):
        self.Warning.performance_warning.clear()
        self.Warning.renamed_vars.clear()
        if self.data is None:
            table = None
        else:
            domain, cols, renamed = \
                self.domain_editor.get_domain(self.data.domain, self.data,
                                              deduplicate=True)
            if not (domain.variables or domain.metas):
                table = None
            elif domain is self.data.domain:
                table = self.data
            else:
                X, y, m = cols
                table = Table.from_numpy(domain, X, y, m, self.data.W)
                table.name = self.data.name
                table.ids = np.array(self.data.ids)
                table.attributes = getattr(self.data, 'attributes', {})
                self._inspect_discrete_variables(domain)
            if renamed:
                self.Warning.renamed_vars(f"Renamed: {', '.join(renamed)}")

        self.Warning.multiple_targets(
            shown=table is not None and len(table.domain.class_vars) > 1)
        self.Outputs.data.send(table)
        self.apply_button.setEnabled(False)

    def get_widget_name_extension(self):
        _, name = os.path.split(self.loaded_file)
        return os.path.splitext(name)[0]

    def send_report(self):
        def get_ext_name(filename):
            try:
                return FileFormat.names[os.path.splitext(filename)[1]]
            except KeyError:
                return "unknown"

        if self.data is None:
            self.report_paragraph("File", "No file.")
            return

        if self.source == self.LOCAL_FILE:
            home = os.path.expanduser("~")
            if self.loaded_file.startswith(home):
                # os.path.join does not like ~
                name = "~" + os.path.sep + \
                       self.loaded_file[len(home):].lstrip("/").lstrip("\\")
            else:
                name = self.loaded_file
            if self.sheet_combo.isVisible():
                name += f" ({self.sheet_combo.currentText()})"
            self.report_items("File", [("File name", name),
                                       ("Format", get_ext_name(name))])
        else:
            self.report_items("Data", [("Resource", self.url),
                                       ("Format", get_ext_name(self.url))])

        self.report_data("Data", self.data)

    @staticmethod
    def dragEnterEvent(event):
        """Accept drops of valid file urls"""
        urls = event.mimeData().urls()
        if urls:
            try:
                FileFormat.get_reader(urls[0].toLocalFile())
                event.acceptProposedAction()
            except MissingReaderException:
                pass

    def dropEvent(self, event):
        """Handle file drops"""
        urls = event.mimeData().urls()
        if urls:
            self.add_path(urls[0].toLocalFile())  # add first file
            self.source = self.LOCAL_FILE


    def workflowEnvChanged(self, key, value, oldvalue):
        """
        Function called when environment changes (e.g. while saving the scheme)
        It make sure that all environment connected values are modified
        (e.g. relative file paths are changed)
        """
        self.update_file_list(key, value, oldvalue)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0
    WidgetPreview(MyWidget).run()
