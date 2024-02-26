import sqlalchemy as sa
from sqlalchemy import inspection
from sqlalchemy.ext import compiler
from sqlalchemy.sql.base import DialectKWArgs
from sqlalchemy.schema import DDLElement, PrimaryKeyConstraint
from sqlalchemy.sql.expression import ClauseElement, Executable
from sqlalchemy.sql.schema import _CreateDropBind, HasSchemaAttr, Table, _get_table_key
from sqlalchemy.sql.ddl import _CreateBase, _DropBase, SchemaGenerator, SchemaDropper

from sqlalchemy_utils.functions import get_columns

class View(Table, inspection.Inspectable["View"]):
    __visit_name__ = "table"  # Behave as table when not used for create/drop view

    def __init__(self,  name, metadata, selectable, materialized=False, replace=False, *args, **kwargs):
        super().__init__(name, metadata, *args, **kwargs)
        if materialized and replace:
            raise ValueError("Cannot use CREATE OR REPLACE with materialized views")
        self._selectable = selectable
        self._materialized = materialized
        self._replace = replace

    
    @property
    def selectable(self):
        return self._selectable

    @selectable.setter
    def selectable(self, value):
        self._selectable = value

    @property
    def materialized(self):
        return self._materialized

    @materialized.setter
    def materialized(self, value):
        self._materialized = value

    @property
    def replace(self):
        return self._replace

    @replace.setter
    def replace(self, value):
        self._replace = value

    def create_view(self, bind: _CreateDropBind, checkfirst: bool = False) -> None:
        """Issue a ``CREATE`` statement for this
        :class:`.Index`, using the given
        :class:`.Connection` or :class:`.Engine`` for connectivity.

        .. seealso::

            :meth:`_schema.MetaData.create_all`.

        """
        View.__visit_name__ = 'view'
        try:
            bind._run_ddl_visitor(ExtSchemaGenerator, self, checkfirst=checkfirst)
        finally:
            View.__visit_name__ = 'table'

    def drop_view(self, bind: _CreateDropBind, checkfirst: bool = False) -> None:
        """Issue a ``DROP`` statement for this
        :class:`.Index`, using the given
        :class:`.Connection` or :class:`.Engine` for connectivity.

        .. seealso::

            :meth:`_schema.MetaData.drop_all`.

        """
        View.__visit_name__ = 'view'
        try:
            bind._run_ddl_visitor(ExtSchemaDropper, self, checkfirst=checkfirst)
        finally:
            View.__visit_name__ = 'table'

    def __repr__(self) -> str:
        return "View(%s)" % ", ".join(
            [repr(self.name)]
            + [repr(self.metadata)]
            + [repr(x) for x in self.columns]
            + ["%s=%s" % (k, repr(getattr(self, k))) for k in ["schema"]]
        )

    def __str__(self) -> str:
        return _get_table_key(self.description, self.schema)

class CreateView(_CreateBase):
    """Represent a CREATE VIEW statement."""

    __visit_name__ = "create_view"

    def __init__(self, element, if_not_exists=False):
        """Create a :class:`.CreateView` construct.

        :param element: a :class:`_schema.View` that's the subject
         of the CREATE.
        :param if_not_exists: if True, an IF NOT EXISTS operator will be
         applied to the construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_not_exists=if_not_exists)


class DropView(_DropBase):
    """Represent a DROP VIEW statement."""

    __visit_name__ = "drop_view"

    def __init__(self, element, if_exists=False):
        """Create a :class:`.DropView` construct.

        :param element: a :class:`_schema.View` that's the subject
         of the DROP.
        :param if_exists: if True, an IF EXISTS operator will be applied to the
         construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_exists=if_exists)


class ExtSchemaGenerator(SchemaGenerator):
    def visit_view(self, view, create_ok=False):
        with self.with_ddl_events(view):
            CreateView(view)._invoke_with(self.connection)

class ExtSchemaDropper(SchemaDropper):
    def visit_view(self, view, drop_ok=False):
        with self.with_ddl_events(view):
            DropView(view)(view, self.connection)


@compiler.compiles(CreateView)
def compile_create_materialized_view(create, compiler, **kw):
    element = create.element
    withclause = element.dialect_options.get("postgresql", {}).get("with", {})
    withclause_text = ''
    if withclause:
        withclause_text += " WITH (%s)" % (
            ", ".join(
                [
                    "%s = %s" % storage_parameter
                    for storage_parameter in withclause.items()
                ]
            )
        )
    return 'CREATE {}{}VIEW {}{} AS {}'.format(
        'OR REPLACE ' if element.replace else '',
        'MATERIALIZED ' if element.materialized else '',
        compiler.dialect.identifier_preparer.quote(element.name),
        withclause_text,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@compiler.compiles(DropView)
def compile_drop_materialized_view(create, compiler, **kw):
    element = create.element
    return 'DROP {}VIEW IF EXISTS {} {}'.format(
        'MATERIALIZED ' if element.materialized else '',
        compiler.dialect.identifier_preparer.quote(element.name),
        'CASCADE' if element.cascade else ''
    )


def create_view_from_selectable(
    name,
    selectable,
    indexes=None,
    metadata=None,
    aliases=None,
    materialized=False,
    replace=False,
    **kwargs
):
    if indexes is None:
        indexes = []
    if metadata is None:
        metadata = sa.MetaData()
    if aliases is None:
        aliases = {}
    args = [
        sa.Column(
            c.name,
            c.type,
            key=aliases.get(c.name, c.name),
            primary_key=c.primary_key
        )
        for c in get_columns(selectable)
    ] + indexes
    view = View(name, metadata, selectable, materialized, replace, *args, **kwargs)

    if not any([c.primary_key for c in get_columns(selectable)]):
        view.append_constraint(
            PrimaryKeyConstraint(*[c.name for c in get_columns(selectable)])
        )
    return view


def create_materialized_view(
    name,
    selectable,
    metadata,
    indexes=None,
    aliases=None,
    **kwargs
):
    """ Create a view on a given metadata

    :param name: The name of the view to create.
    :param selectable: An SQLAlchemy selectable e.g. a select() statement.
    :param metadata:
        An SQLAlchemy Metadata instance that stores the features of the
        database being described.
    :param indexes: An optional list of SQLAlchemy Index instances.
    :param aliases:
        An optional dictionary containing with keys as column names and values
        as column aliases.

    Same as for ``create_view`` except that a ``CREATE MATERIALIZED VIEW``
    statement is emitted instead of a ``CREATE VIEW``.

    """
    view = create_view_from_selectable(
        name=name,
        selectable=selectable,
        indexes=indexes,
        metadata=None,
        aliases=aliases,
        materialized=True,
        replace=False,
        **kwargs
    )

    @sa.event.listens_for(metadata, 'after_create')
    def create_view(target, connection, **kw):
        view.create_view(connection)

    @sa.event.listens_for(metadata, 'after_create')
    def create_indexes(target, connection, **kw):
        for idx in view.indexes:
            idx.create(connection)

    @sa.event.listens_for(metadata, 'before_drop')
    def drop_view(target, connection, **kw):
        view.drop_view(connection)

    return view


def create_view(
    name,
    selectable,
    metadata,
    cascade_on_drop=True,
    replace=False,
    **kwargs
):
    """ Create a view on a given metadata

    :param name: The name of the view to create.
    :param selectable: An SQLAlchemy selectable e.g. a select() statement.
    :param metadata:
        An SQLAlchemy Metadata instance that stores the features of the
        database being described.
    :param cascade_on_drop: If ``True`` the view will be dropped with
        ``CASCADE``, deleting all dependent objects as well.
    :param replace: If ``True`` the view will be created with ``OR REPLACE``,
        replacing an existing view with the same name.

    The process for creating a view is similar to the standard way that a
    table is constructed, except that a selectable is provided instead of
    a set of columns. The view is created once a ``CREATE`` statement is
    executed against the supplied metadata (e.g. ``metadata.create_all(..)``),
    and dropped when a ``DROP`` is executed against the metadata.

    To create a view that performs basic filtering on a table. ::

        metadata = MetaData()
        users = Table('users', metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String),
                Column('fullname', String),
                Column('premium_user', Boolean, default=False),
            )

        premium_members = select(users).where(users.c.premium_user == True)
        # sqlalchemy 1.3:
        # premium_members = select([users]).where(users.c.premium_user == True)
        create_view('premium_users', premium_members, metadata)

        metadata.create_all(engine) # View is created at this point

    """
    view = create_view_from_selectable(
        name=name,
        selectable=selectable,
        metadata=None,
        replace=replace,
        **kwargs
    )

    @sa.event.listens_for(metadata, 'after_create')
    def create_view(target, connection, **kw):
        view.create_view(connection)

    @sa.event.listens_for(metadata, 'after_create')
    def create_indexes(target, connection, **kw):
        for idx in view.indexes:
            idx.create(connection)

    @sa.event.listens_for(metadata, 'before_drop')
    def drop_view(target, connection, **kw):
        view.drop_view(connection)

    return view


class RefreshMaterializedView(Executable, ClauseElement):
    inherit_cache = True

    def __init__(self, name, concurrently):
        self.name = name
        self.concurrently = concurrently


@compiler.compiles(RefreshMaterializedView)
def compile_refresh_materialized_view(element, compiler):
    return 'REFRESH MATERIALIZED VIEW {concurrently}{name}'.format(
        concurrently='CONCURRENTLY ' if element.concurrently else '',
        name=compiler.dialect.identifier_preparer.quote(element.name),
    )


def refresh_materialized_view(session, name, concurrently=False):
    """ Refreshes an already existing materialized view

    :param session: An SQLAlchemy Session instance.
    :param name: The name of the materialized view to refresh.
    :param concurrently:
        Optional flag that causes the ``CONCURRENTLY`` parameter
        to be specified when the materialized view is refreshed.
    """
    # Since session.execute() bypasses autoflush, we must manually flush in
    # order to include newly-created/modified objects in the refresh.
    session.flush()
    session.execute(RefreshMaterializedView(name, concurrently))
