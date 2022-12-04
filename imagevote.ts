import { GImage, GImageSet, ColumnHeader, FIELDS, sort, createElement } from "./types.js"
import { loadImageSets } from "./build.js"
import { StoredVal } from "./storage.js"
import { GImageGrid } from "./grid.js"
import { GridHeaders } from "./grid_headers.js"

var grid: GImageGrid

const STORE_VOTES = new StoredVal('image_votes', new Map<string, number>(), 
                                    storage => Array.from(storage.entries()), jsonVal => new Map(jsonVal as any[]))
function onclick(filename: string) {
    var image = grid.imageByFilename.get(filename)
    if (image == null) {
        console.log(`onclickThumbnail: logic error: can't find image with filename ${filename}`)
        return
    }

    image.votes ++
    STORE_VOTES.get().set(filename, image.votes)
    STORE_VOTES.save()

    renderVoteStats()
    renderNext()
}

function renderNext() {
    const width = 3, height = 2
    const wantedCount = width * height
    const filenames = Array.from(grid.imageByFilename.keys())

    var picks = new Array<GImage>()

    while (picks.length < wantedCount && filenames.length >= picks.length) {
        const idx = Math.floor(Math.random() * filenames.length)
        picks.push(grid.imageByFilename.get(filenames[idx])!)
    }

    var gridElem = document.getElementById("grid")
    gridElem!.innerHTML = ""

    for (var y = 0; y < height; y ++) {
        for (var x = 0; x < width; x ++) {
            const image = picks.pop()!
            const imageElem = createElement('img', {'src': image.filename })
            imageElem.style.gridColumn = (x + 1).toString()
            imageElem.style.gridRow = (y + 1).toString()
            imageElem.onclick = function (ev) { onclick(image.filename) }
            gridElem!.appendChild(imageElem)
        }
    }

}

function renderVoteStats() {
    var votesElem = document.getElementById("vote_results")
    if (votesElem == null) {
        console.log("votesElem not found")
        return
    }

    var html = ""
    for (const field of FIELDS) {
        const votes = new Map<string, number>() // value to count
        var count = 0

        for (const filename of STORE_VOTES.get().keys()) {
            const image = grid.imageByFilename.get(filename)
            if (image == null) {
                console.log(`renderVoteStats: ${filename} not found`)
                continue
            }
            const iset = grid.imagesetByFilename.get(filename)!
            const key = iset[field]
            if (votes.has(key)) {
                count = votes.get(key)!
            }
            count ++
            votes.set(key, count)
        }

        var values = Array.from(votes.keys()).sort((a, b) => {
            var aval = votes.get(a)!
            var bval = votes.get(b)!
            return bval - aval
        })
    
        html += `${field}:<br/>\n`
        for (const value of values) {
            const count = votes.get(value)!
            html += `&nbsp;&nbsp;${value}: ${count}<br/>\n`
        }
    }

    votesElem.innerHTML = html
}


async function loadImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        const imageSets = loadImageSets(filenames)
        grid = new GImageGrid(imageSets)

        const filenamesVotes = STORE_VOTES.get()
        for (const [filename, count] of filenamesVotes.entries()) {
            const image = grid.imageByFilename.get(filename)
            if (image) {
                image.votes = count
            }
        }
    }
    else {
        console.log("error")
    }
}

loadImages().then((val) => {
    console.log("loaded images.")

    renderVoteStats()
    renderNext()
})
