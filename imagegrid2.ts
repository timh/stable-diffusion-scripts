import { GImage, GImageSet, ColumnHeader, FIELDS, sort, createElement } from "./types.js"
import { loadImageSets } from "./build.js"
import { StoredVal } from "./storage.js"
import { GImageGrid } from "./grid.js"
import { GridHeaders } from "./grid_headers.js"

var grid: GImageGrid
var gridHeaders: GridHeaders

function onclickChoice(field: string, choice: any): any {
    var visibility = grid.setVisibility(field, choice, "toggle")

    // if toggling a modelName, also toggle the modelStr's that are subsets of it.
    if (field == 'modelName' || field == 'modelSeed' || field == 'modelSteps') {
        var matchingModelStrs = grid.fieldUniqueValues.get('modelStr') as Array<string>
        for (const [matchIdx, matchChoice] of matchingModelStrs.entries()) {
            if (field == 'modelName' && matchChoice.startsWith(choice as string)) {
                // modelStr that starts with this modelName should be matched.
            }
            else if (field == 'modelSeed') {
                // modelStr that has r{seed} in it should match.
                var seedStr = ` r${choice} `
                if (matchChoice.indexOf(seedStr) == -1) {
                    continue;
                }
            }
            else if (field == 'modelSteps' && matchChoice.endsWith(choice as string)) {
                // modelStr that ends with _{steps} should match
            }
            else {
                // everything else shouldn't match.
                continue;
            }

            grid.setVisibility('modelStr', matchChoice, visibility)
        }
    }
}

function renderAllChoices() {
    for (const field of FIELDS) {
        renderChoices(field)
    }
}

function renderChoices(field: string) {
    var choices = grid.fieldUniqueValues.get(field)!
    var chooserDiv = document.getElementById('chooser')!

    var span = document.createElement("span")
    span.className = "field"
    span.appendChild(document.createTextNode(field))
    chooserDiv.appendChild(span)

    span = document.createElement("span")
    span.className = "values"
    chooserDiv.appendChild(span)

    for (const [idx, choice] of choices.entries()) {
        var choiceSpan = document.createElement("span")
        var fieldClass = `${field}_${idx}`
        choiceSpan.className = "selected"
        choiceSpan.id = `choice_${fieldClass}`

        choiceSpan.onclick = function(this: GlobalEventHandlers, ev: MouseEvent): any {
            onclickChoice(field, choice)
        }
        choiceSpan.appendChild(document.createTextNode(choice.toString()))
        span.appendChild(choiceSpan)
    }
}

const STORE_CHECKED = new StoredVal('images_selected', new Set<string>(), storage => Array.from(storage), jsonVal => new Set(jsonVal))
function onclickThumbnail(ev: MouseEvent, filename: string) {
    var image = grid.imageByFilename.get(filename)
    if (image == null) {
        console.log(`onclickThumbnail: logic error: can't find image with filename ${filename}`)
        return
    }
    var isChecked = image.checked
    var newChecked = !isChecked

    var imgElement = ev.target as HTMLElement
    var selectElem = imgElement.parentElement?.getElementsByClassName("image_select")?.item(0)
    if (selectElem == null) {
        console.log(`onclickThumbnail: logic error: can't find image_select span for filename ${filename}`)
        return
    }

    image.checked = newChecked
    if (newChecked) {
        selectElem.className += " checked"
        STORE_CHECKED.get().add(filename)
    }
    else {
        selectElem.className = selectElem.className.replace(" checked", "")
        STORE_CHECKED.get().delete(filename)
    }
    STORE_CHECKED.save()
    renderCheckStats()
}

function renderCheckStats() {
    var resultsElem = document.getElementById("checked_results")
    if (resultsElem == null) {
        console.log("resultsElem not found")
        return
    }

    var html = ""

    for (const field of FIELDS) {
        var fieldStats = new Map<any, number>()
        for (const filename of STORE_CHECKED.get()) {
            var iset = grid.imagesetByFilename.get(filename)
            if (iset == null) {
                console.log(`renderCheckStats: can't find imageset for filename ${filename}`)
                continue
            }
            const fieldVal = iset[field]
            var curCount = 0
            if (!fieldStats.has(fieldVal)) {
                fieldStats.set(fieldVal, 0)
            }
            else {
                curCount = fieldStats.get(fieldVal)!
            }
            fieldStats.set(fieldVal, curCount + 1)
        }
        var values = Array.from(fieldStats.keys()).sort((a, b) => {
            var aval = fieldStats.get(a)!
            var bval = fieldStats.get(b)!
            return bval - aval
        })
        html += `${field}:<ul/>\n`
        for (const [idx, value] of values.entries()) {
            html += "<li>"
            if (field == 'prompt') {
                html += `"${value}"`
            }
            else {
                html += value.toString()
            }
            html += `: ${fieldStats.get(value)}</li>\n`
        }
        html += "</ul>\n"
    }
    resultsElem.innerHTML = html
}

function renderGridHeaders() {
    var gridElem = document.getElementById("imagegrid") as HTMLElement
    gridElem.innerHTML = ""
    gridHeaders.headers.forEach((header) => {
        var span = createElement('span', {'class': header.classes})
        span.style.gridRow = header.row.toString()
        span.style.gridColumnStart = header.columnStart.toString()
        span.style.gridColumnEnd = header.columnEnd.toString()
        span.textContent = header.value
        gridElem.appendChild(span)
    })

    // generate row labels for all the seeds
    var allSeedsSet = new Set<number>()
    for (const iset of grid.imageSets.values()) {
        for (const img of iset.images) {
            allSeedsSet.add(img.seed)
        }
    }
    var allSeeds = sort(allSeedsSet)
    for (const [idx, seed] of allSeeds.entries()) {
        var span = createElement('span', {}, seed.toString())
        span.style.gridRow = (idx + FIELDS.length + 1).toString()
        span.style.gridColumn = "1"
        gridElem.appendChild(span)
    }        
}

function renderGridImages() {
    var gridElem = document.getElementById("imagegrid") as HTMLElement

    for (const [isetIdx, setKey] of grid.imageSetKeys.entries()) {
        var iset = grid.imageSets.get(setKey) as GImageSet
        var column = isetIdx + 2
        var classes = FIELDS.map((field) => {
            var val = iset[field]
            return `${field}_${grid.fieldValueIndex.get(field)?.get(val)}`
        }).join(" ")

        for (const [imageIdx, image] of iset.images.entries()) {
            var row = imageIdx + FIELDS.length + 1
            
            var topSpan = createElement('span', {'class': `image ${classes}`})
            topSpan.style.gridRow = row.toString()
            topSpan.style.gridColumn = column.toString()
            var selectElem = topSpan.appendChild(createElement('span', {'class': 'image_select'}, "checked"))
            if (image.checked) {
                selectElem.className += " checked"
            }

            var thumbElem = topSpan.appendChild(createElement('img', {'src': image.filename, 'class': "thumbnail"}))
            thumbElem.onclick = function(this, ev) {
                onclickThumbnail(ev, image.filename)
            }

            var detailsSpan = topSpan.appendChild(createElement('span', {'class': "details"}))
            var imageElem = detailsSpan.appendChild(createElement('img', {'src': image.filename, 'class': "fullsize"}))
            var detailsGrid = detailsSpan.appendChild(createElement('div', {'class': "details_grid"}))

            var entries = {"model": iset.modelStr, "prompt": iset.prompt, 
                           "sampler": `${iset.sampler} ${iset.samplerSteps}`,
                           "CFG": iset.cfg.toString(), "seed": image.seed.toString()}
            for (const key in entries) {
                const value = entries[key]
                var keySpan = createElement('span', {'class': "detailsKey"})
                keySpan.textContent = key
                var valueSpan = createElement('span', {'class': "detailsVal"})
                valueSpan.textContent = value
                detailsGrid.appendChild(keySpan)
                detailsGrid.appendChild(valueSpan)
            }

            gridElem.appendChild(topSpan)
        }
    }

}

async function loadImages() {
    var resp = await fetch("filelist.txt");
    
    const data = await resp.text()
    if (resp.ok) {
        var filenames = data.split("\n")

        const imageSets = loadImageSets(filenames)
        grid = new GImageGrid(imageSets)
        gridHeaders = new GridHeaders(grid)

        const filenamesSelected = STORE_CHECKED.get()
        for (const filename of filenamesSelected) {
            const image = grid.imageByFilename.get(filename)
            if (image) {
                image.checked = true
            }
        }
    }
    else {
        console.log("error")
    }
}

loadImages().then((val) => {
    console.log("loaded images.")

    console.log("renderAllChoices")
    renderAllChoices()
    console.log("renderCheckStats")
    renderCheckStats()

    console.log("renderGridHeaders")
    renderGridHeaders()
    console.log("renderGridImages")
    renderGridImages()

    grid.loadVisibilityFromStore()
})
