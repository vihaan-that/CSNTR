
import click
import sys
import os

# Add the directory where retrievalAPI.py and indexerAPI.py are located
sys.path.append(os.path.abspath('../API'))

from retrievalAPI import find
from indexerAPI import createNote, editNote, index, deleteNote

@click.group()
def cli():
    """A command-line interface for managing notes and querying documents."""
    pass

# Command to create a new note
@cli.command()
@click.argument('filename')
def create(filename):
    """Create a new note with FILENAME."""
    try:
        createNote(filename)
        click.echo(f"Note '{filename}' created successfully.")
    except FileExistsError as e:
        click.echo(str(e))

# Command to edit an existing note
@cli.command()
@click.argument('filename')
@click.argument('text')
def edit(filename, text):
    """Edit the note with FILENAME by replacing its content with TEXT."""
    try:
        editNote(filename, text)
        click.echo(f"Note '{filename}' edited successfully.")
    except FileNotFoundError as e:
        click.echo(str(e))

# Command to index a note
@cli.command()
@click.argument('filename')
def index_note(filename):
    """Index the note with FILENAME."""
    try:
        index(filename)
        click.echo(f"Note '{filename}' indexed successfully.")
    except FileNotFoundError as e:
        click.echo(str(e))

# Command to delete a note
@cli.command()
@click.argument('filename')
def delete(filename):
    """Delete the note with FILENAME."""
    try:
        deleteNote(filename)
        click.echo(f"Note '{filename}' and its embedding deleted successfully.")
    except FileNotFoundError as e:
        click.echo(str(e))

# Command to query documents
@cli.command()
@click.argument('query_text')
def query(query_text):
    """Find the top 3 most similar documents for the QUERY_TEXT."""
    results = find(query_text)
    if results:
        click.echo(f"\nTop 3 results for the query: {query_text}")
        for idx, (doc_name, similarity, content) in enumerate(results):
            click.echo(f"\nRank {idx + 1}:\nDocument: {doc_name}\nSimilarity: {similarity:.4f}\nContent:\n{content}\n")
    else:
        click.echo("No similar documents found.")

if __name__ == "__main__":
    cli()
